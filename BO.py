from botorch.models import SingleTaskGP, MixedSingleTaskGP, SingleTaskMultiFidelityGP
from botorch.acquisition import UpperConfidenceBound, LogExpectedImprovement, PosteriorMean, qKnowledgeGradient
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf, optimize_acqf_mixed_alternating
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms import Normalize

import time
import torch
import numpy as np
import os
from tqdm import tqdm
from typing import List

from transformers import TrainerCallback
from datasets import concatenate_datasets
from LLM.llm import sample, tokenizing_method, train_on_inputs, add_eos_token, evaluate_tasks, get_tokenizer_and_model, load_data, extract_data_mixture_and_train
from LLM.tokenize_util import tokenizing_method
from BO_utils.acquisition_functions import CostScaledLogEI, CostScaledUCB, CostScaledKG, cost_fn
from BO_utils.utils import print_non_lora_params, print_lora_params
from BO_utils.utils import randomly_generate_data, print_inputs, arrange_lora_config, process_candidate, inverse_process_candidate, generate_bounds

os.environ["HF_ALLOW_CODE_EXECUTION"] = "1"

from peft import (
    LoraConfig,
    get_peft_model,
)
       
def to_serializable(val: float | int | torch.Tensor | np.floating | np.integer) -> float | int:
    """Convert a value to a JSON-serializable Python float or int.

    Handles torch Tensors (scalar), numpy floating/integer types, and
    native Python float/int. Returns the value unchanged if none of
    the known types match.

    Args:
        val: A scalar value that may be a Tensor, numpy type, or native Python type.

    Returns:
        A plain Python float or int suitable for JSON serialization.
    """
    if isinstance(val, torch.Tensor):
        val = val.item()
    return float(val) if isinstance(val, float) or isinstance(val, np.floating) else int(val) if isinstance(val, int) or isinstance(val, np.integer) else val

def get_lora_and_mixing_ratio(
    input_X: List[float],
    what_to_optimize: str,
    data_domains: List[str],
    lora_max_num_layers: int,
    default_lora_config: LoraConfig,
    fidelity: int | None
) -> tuple[LoraConfig | None, List[float], dict]:
    """Derive the LoRA config, data mixing ratio, and discrete-dim map from a candidate input X.

    Depending on ``what_to_optimize``, either uses defaults or extracts values
    from ``input_X`` to build the LoRA configuration and the per-domain
    mixing ratio.  Also returns the ``discrete_dims`` dict required by
    ``MixedSingleTaskGP`` / ``optimize_acqf_mixed_alternating``.

    Args:
        input_X: Candidate parameter vector (actual values).
        what_to_optimize: One of ``"data"``, ``"model"``, or ``"both"``.
        data_domains: Data domain names.
        lora_max_num_layers: Total number of layers in the base model.
        default_lora_config: Fallback LoRA config used when only data is optimised.
        fidelity: Current fidelity value; if not ``None``, fidelity is added to
            ``discrete_dims``.

    Returns:
        lora_config: The constructed (or default) ``LoraConfig``.
        mixing_ratio: Per-domain data mixing ratios.
        discrete_dims: Mapping from dimension index to allowed discrete values.
    """
    # Build LoRA config if optimizing LoRA
    if what_to_optimize == "data":
        lora_config = default_lora_config # default LoRA configuration
        mixing_ratio = input_X
        discrete_dims = {}
    elif what_to_optimize == "model":
        
        discrete_dims = {
                1: [0,1], # modules
                2: [0,1], # module 2
                3: [0,1], # module 3
                4: [0,1], # module 4
                5: [0,1], # module 5
                9: [0,1], # whether to apply LoRA in reverse
            }
        
        mixing_ratio = [1/len(data_domains)]*len(data_domains) # default using uniform mixture
        idx=0
        lora_config = arrange_lora_config(
            lora_r=input_X[-4],
            lora_dropout=input_X[-3],
            num_layers_to_apply=input_X[idx],
            five_dim_vector=input_X[idx+1:idx+1+5],
            lora_alpha = input_X[-2],
            lora_reverse = input_X[-1],
            max_num_layers = lora_max_num_layers
        )
        
    elif what_to_optimize == "both":
        # specifies the discrete dimension
        discrete_dims = {
                len(data_domains)+1: [0,1], # modules
                len(data_domains)+2: [0,1], # module 2
                len(data_domains)+3: [0,1], # module 3
                len(data_domains)+4: [0,1], # module 4
                len(data_domains)+5: [0,1], # module 5
                len(data_domains)+9: [0,1] # whether to apply LoRA in reverse
            }
        idx = len(data_domains)
        mixing_ratio = input_X[:idx]
        lora_config = arrange_lora_config(
            lora_r=input_X[-4],
            lora_dropout=input_X[-3],
            num_layers_to_apply=input_X[idx],
            five_dim_vector=input_X[idx+1:idx+1+5],
            lora_alpha = input_X[-2],
            lora_reverse = input_X[-1],
            max_num_layers = lora_max_num_layers,
        )
    else:
        assert False, "what_to_optimize is not properly set."

    if fidelity is not None:
        discrete_dims[len(input_X)] = [0,1] # dimension indicating fidelity is discrete too
    
    return lora_config, mixing_ratio, discrete_dims

def evaluate_single_configuration(
                input_X: List[float],
                fidelity: int | None,
                what_to_optimize: str,
                data_domains: List[str],
                lora_max_num_layers: int,
                lora_rank_max: int,
                default_lora_config: LoraConfig,
                model_id: str,
                train_datasets: list,
                val_datasets: list,
                all_sampled_evaluation_data,
                total_number_datapoints: int,
                sampling_method: str,
                train_epochs: int,
                training_batch: int,
                max_steps: int,
                eval_steps: int,
                evaluation_task: dict,
                eval_method: str,
                evaluation_batch: int,
                num_eval_samples: int,
                time_callback: TrainerCallback,
                seed: int = 42,
            ) -> tuple[float, float, dict]:
                """
                Evaluates a single configuration (input_X) at a given fidelity level.
                
                Args:
                    input_X: List of parameter values (data mixing ratios and/or LoRA params)
                    fidelity: Fidelity level (0 for low, 1 for high, None for single-fidelity)
                    ... (other parameters as needed for training and evaluation)
                
                Returns:
                    observed_performance: Performance observed at the specified fidelity
                    realized_performance: Full training performance (always high-fidelity)
                    train_results: Dictionary containing training metrics and logs
                """
                print(f"\n=== Evaluating Configuration ===")
                print(f"Fidelity level: {fidelity}")
                
                # Setup model and tokenizer
                torch.manual_seed(seed)
                np.random.seed(seed)
                tokenizer, model = get_tokenizer_and_model(model_id=model_id)
                
                # Print configuration
                print_inputs(input_X, data_domains, what_to_optimize)
                
                # Get LoRA config and mixing ratio
                lora_config, mixing_ratio, _ = get_lora_and_mixing_ratio(
                    input_X, what_to_optimize, data_domains, lora_max_num_layers, 
                    default_lora_config, fidelity
                )
                
                # Apply LoRA to model
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
                
                # Setup performance callback if needed
                callbacks = [time_callback]
                if eval_method == "performance":
                    class PerformanceEvalCallback(TrainerCallback):
                        def __init__(self, tokenizer, eval_tasks):
                            self.tokenizer = tokenizer
                            self.target_eval_steps = {25, 50, 75, 100}
                            self.eval_tasks = eval_tasks
                            self.step_performances = {}
                        
                        def _evaluate_performance(self, model):
                            model.eval()
                            results = evaluate_tasks(
                                list(self.eval_tasks.keys()), model, self.tokenizer, 
                                batch=8, few_shot=3, limit=num_eval_samples
                            )
                            model.train()
                            
                            for task, value in self.eval_tasks.items():
                                _, metric = value
                                performance = results["results"][task][metric]
                                break
                            return performance
                        
                        def on_step_end(self, _args, state, _control, **kwargs):
                            if state.global_step in self.target_eval_steps:
                                performance = self._evaluate_performance(
                                    kwargs.get('model')
                                )
                                key_name = f"performance_step_{state.global_step}"
                                self.step_performances[key_name] = performance
                    
                    performance_eval_callback = PerformanceEvalCallback(tokenizer, evaluation_task)
                    callbacks.append(performance_eval_callback)
                
                # Train model
                train_results = extract_data_mixture_and_train(
                    model=model,
                    tokenizer=tokenizer,
                    train_datasets=train_datasets,
                    val_datasets=val_datasets,
                    data_domains=data_domains,
                    evaluation_dataset=all_sampled_evaluation_data,
                    mixing_ratio=mixing_ratio,
                    additional_info=[None for _ in data_domains],
                    total_number_datapoints=total_number_datapoints,
                    method=sampling_method,
                    train_epochs=train_epochs,
                    batch_size=training_batch,
                    max_step=max_steps,
                    lora_config=lora_config,
                    eval_steps=eval_steps,
                    callback=callbacks,
                    seed=seed
                )
                
                # Evaluate performance
                observed_performance, realized_performance = evaluate_final_performance(
                    model, tokenizer, eval_method, fidelity, evaluation_task, 
                    train_results, evaluation_batch, num_eval_samples
                )
                
                print(f"Observed performance: {observed_performance}")
                print(f"Realized performance: {realized_performance}")
                
                return observed_performance, realized_performance, train_results

def evaluate_final_performance(
    model: torch.nn.Module,
    tokenizer,
    eval_method: str,
    fidelity: int | None,
    evaluation_task: dict,
    train_results: dict,
    evaluation_batch: int,
    num_eval_samples: int
) -> tuple[float, float]:
    """Compute observed and realised performance after training.

    For ``eval_method="performance"``, runs the evaluation harness tasks and
    uses the best callback-recorded performance as the observed value.
    For ``eval_method="eval_loss"``, uses the (negated) minimum eval loss,
    with low-fidelity returning only the first-half trajectory minimum.

    Args:
        model: The trained model to evaluate.
        tokenizer: Tokenizer associated with the model.
        eval_method: ``"performance"`` or ``"eval_loss"``.
        fidelity: Fidelity level (0 = low, 1 = high, ``None`` = single-fidelity).
        evaluation_task: Dict mapping task name to ``(weight, metric)`` tuples.
        train_results: Training output dict containing ``"eval_loss"`` and/or
            ``"step_performances"``.
        evaluation_batch: Batch size for evaluation.
        num_eval_samples: Number of evaluation samples (0 means unlimited).

    Returns:
        observed_performance: What the GP observes (may be low-fidelity).
        realized_performance: True end-of-training performance. This is used for plotting purpose to evaluate how well the BO is performing.
    """
    model.eval()
    observed_performance = None # what is observed, possibly low-fidelity.
    realized_performance = None # performance at the end of training. Potentially the same as observed_performance.
    if eval_method == "performance":
        print("evaluating with task performance...")
        if num_eval_samples == 0:
            num_eval_samples = None
        results = evaluate_tasks(list(evaluation_task.keys()), model, tokenizer, evaluation_batch, few_shot=5, limit=num_eval_samples)
        perf = 0
        for task, (weight, metric) in evaluation_task.items():
            p = results["results"][task][metric]
            if task=="wikitext": p=-p
            perf += p*weight
        observed_performance =  max(train_results["step_performances"].values()) # from callback
        realized_performance = perf
    elif eval_method == "eval_loss": # if we want to minimize loss
        if fidelity == 0: # if lower fidelity is queried, 
            eval_loss_trajectory = train_results["eval_loss"]

            half = len(eval_loss_trajectory) // 2          # first 50%; TODO: parameterized this
            observed_performance = - min(eval_loss_trajectory[:half])
            realized_performance = - min(train_results["eval_loss"])
        else:
            observed_performance = - min(train_results["eval_loss"])
            realized_performance = - min(train_results["eval_loss"])
    else:
        assert False, "eval_method not properly set."
        
    return observed_performance, realized_performance

def fit_GP_and_suggest_next_candidate(
    GP_input: List[List[float]], 
    observed_output: List[float], 
    fidelity: int | None, 
    what_to_optimize: str, 
    BO_params: dict, 
    max_performance_so_far: float, 
    bounds: torch.Tensor, 
    cost_fn, 
    data_domains: List[str], 
    discrete_dims: dict, 
    itr: int, 
    lora_max_num_layers: int, 
    lora_rank_max: int
) -> tuple[torch.Tensor, int | None, SingleTaskGP | MixedSingleTaskGP | SingleTaskMultiFidelityGP]:
    """Fit a Gaussian Process to observations and propose the next candidate.

    Selects the appropriate GP type (single-task, mixed, or multi-fidelity),
    fits it via maximum likelihood, constructs the acquisition function
    (UCB, EI, cost-scaled UCB, or Knowledge Gradient), optimises it subject
    to equality / inequality constraints, and returns the best candidate.

    Args:
        GP_input: List of observed normalised input vectors.
        observed_output: Corresponding observed objective values.
        fidelity: Current fidelity setting (``None`` for single-fidelity).
        what_to_optimize: One of ``"data"``, ``"model"``, or ``"both"``.
        BO_params: BO config dict (keys: ``"optimize_method"``, ``"acq_function"``,
            ``"ucb_beta"``, etc.).
        max_performance_so_far: Best objective value seen so far (for EI).
        bounds: ``(2, d)`` tensor of lower and upper bounds.
        cost_fn: Cost function for multi-fidelity acquisition functions.
        data_domains: Data domain names.
        discrete_dims: Dict mapping dimension index → allowed discrete values.
        itr: Current BO iteration (used for UCB beta schedule).
        lora_max_num_layers: Max layers scaling factor.
        lora_rank_max: Max rank scaling factor.

    Returns:
        candidate: ``(1, d)`` tensor of the proposed next normalised candidate.
        next_fidelity: Suggested fidelity level, or ``None``.
        gp: The fitted GP model.
    """
    # fit GP
    if fidelity is not None:
        print("fitting a GP: SingleTaskMultiFidelityGP")
        gp = SingleTaskMultiFidelityGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), data_fidelities=[len(GP_input[0])-1], outcome_transform=Standardize(m=1), input_transform=Normalize(d=len(GP_input[0])))
    else:
        if what_to_optimize == "data":
            print("Because we only have continuous inputs, we fit a normal GP")
            gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1), input_transform=Normalize(d=len(GP_input[0])))
        else:
            print("Because we have discrete inputs, we fit a MixedSingleTaskGP")
            gp = MixedSingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), discrete_dims, outcome_transform=Standardize(m=1), input_transform=Normalize(d=len(GP_input[0])))
    
    # maximum likelihood estimation to fit the GP
    if BO_params["optimize_method"] != "random":
        print("performing maximum likelihood estimation to fit the GP...")
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

    # initialize acquisition function
    if BO_params["acq_function"] == "ucb":
        acq = UpperConfidenceBound(gp, beta=BO_params["ucb_beta"]/(2*(itr+1)**0.5))
    if BO_params["acq_function"] == "EI":
        acq = LogExpectedImprovement(gp, best_f=max_performance_so_far)
    if BO_params["optimize_method"] == "multi_fidelity":
        acq = CostScaledUCB(model=gp, beta=BO_params["ucb_beta"]/(2*(itr+1)**0.5), cost_fn=cost_fn)
    if BO_params["optimize_method"] == "multi_fidelity_KG":
        print("building KG acq function")
        num_fantasies = 8
        # base KG
        qKG = qKnowledgeGradient(gp, num_fantasies=num_fantasies)
    
        # get current best posterior mean
        argmax_pmean, max_pmean = optimize_acqf(
            acq_function=PosteriorMean(gp),
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        
        acq = CostScaledKG(model=gp, cost_fn=cost_fn, num_fantasies=num_fantasies, current_max_pmean=max_pmean, sampler=qKG.sampler)
    
    next_fidelity = None
    candidate = None
    equality_constraints = None
    inequality_constraints = None

    # Define equality constraints on data dimensions (if exists, it is the first len(data_domains) dim)
    # mixing ratio must sum to 1
    if what_to_optimize == "data":
        d = len(data_domains)
        equality_constraints = [
            (
                torch.arange(d, dtype=torch.long),
                torch.ones(d, dtype=torch.double),
                1.0
            )
        ]
        
    if what_to_optimize == "model":
        
        # hardcoded - indices 1 to 5 represent which module to apply LoRA
        indices = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        coeffs = torch.ones(5, dtype=torch.double)

        # at least one module should be applied with LoRA
        inequality_constraints = [
            (indices, coeffs, 1.0)
        ]
    
    elif what_to_optimize == "both":
        d = len(data_domains)
        equality_constraints = [
            (
                torch.arange(d, dtype=torch.long),
                torch.ones(d, dtype=torch.double),
                1.0
            )
        ]
        
        # hardcoded - indices 1 to 5 represent which module to apply LoRA
        indices = torch.tensor([len(data_domains)+1, len(data_domains)+2, len(data_domains)+3, len(data_domains)+4, len(data_domains)+5], dtype=torch.long)
        coeffs = torch.ones(5, dtype=torch.double)

        # at least one module should be applied with LoRA
        inequality_constraints = [
            (indices, coeffs, 1.0)
        ]

    if BO_params["optimize_method"] in ["continuous_relaxation", "mixed", "multi_fidelity", "multi_fidelity_KG"]:

        if BO_params["optimize_method"] == "continuous_relaxation":
            candidate, _ = optimize_acqf(
                acq, bounds=bounds, q=1, num_restarts=5, raw_samples=1024, equality_constraints=equality_constraints, inequality_constraints=inequality_constraints
            )
            
            return candidate, next_fidelity, gp

        elif BO_params["optimize_method"] == "mixed":
            if what_to_optimize == "data":
                print("Using default optimize_acqf with constraints for continuous variables")
                print("bounds: ",bounds)
                candidate, _ = optimize_acqf(
                acq, bounds=bounds, q=1, num_restarts=20, raw_samples=1024, equality_constraints=equality_constraints, inequality_constraints=inequality_constraints
                )
            elif what_to_optimize in ["model", "both"]:
                candidate, _ = optimize_acqf_mixed_alternating(
                acq, bounds=bounds, q=1, num_restarts=20, raw_samples=1024, discrete_dims=discrete_dims,
                equality_constraints=equality_constraints if what_to_optimize == "both" else None, inequality_constraints=inequality_constraints
                )
            else:
                assert False, "what_to_optimize not properly set for mixed optimization"
            
            return candidate, next_fidelity, gp

        elif BO_params["optimize_method"] == "multi_fidelity":
            candidate, _ = optimize_acqf_mixed_alternating(
                acq, bounds=bounds, q=1, num_restarts=20, raw_samples=1024, discrete_dims=discrete_dims,
                equality_constraints=equality_constraints, inequality_constraints=inequality_constraints
            )
            
            # split the candidate into the candidate inputs and fidelity
            # TODO: assertion check that candidate has correct dimension (since we didn't do fidelity None check)
            next_fidelity = round(candidate[0][-1].item())
            candidate = candidate[:, :-1]
            return candidate, next_fidelity, gp

        elif BO_params["optimize_method"] == "multi_fidelity_KG":
            t_prev = time.time()
            q = 1 + 8  # for KG, we need to add one more batch
            print("Optimzing KG acquisition function, which is more computationally expensive since it involves fantasizing. This may take a while...")
            candidate, acq_value = optimize_acqf(
                acq_function=acq, bounds=bounds, q=q, num_restarts=10, raw_samples=512,
                equality_constraints=equality_constraints, inequality_constraints=inequality_constraints
            )
            candidate = candidate[0:1, :] # this is not a bug. we only take first entry because KG returns a 1 + 64 batches.
            t_now = time.time()
            print(f"Time taken to perform multi_fidelity_KG optimization: {t_now - t_prev:.4f} seconds")
    
            # split the candidate into the candidate inputs and fidelity
            # TODO: assertion check that candidate has correct dimension (since we didn't do fidelity None check)
            next_fidelity = round(candidate[0][-1].item())
            candidate = candidate[:, :-1]
            return candidate, next_fidelity, gp
    
    elif BO_params["optimize_method"] == "random":
        print("acq optimization method is random sampling")
        _, candidate, next_fidelity = randomly_generate_data(what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max, BO_params=BO_params, seed=None) # fidelty is None here for sure
        candidate = torch.tensor([candidate], dtype=torch.double) # convert to tensor
        return candidate, next_fidelity, gp
    else:
        assert False, "optimize_method not properly set."

def joint_opt_BO_LLM_generalized(
    default_lora_config : LoraConfig,
    time_callback : object,
    lora_rank_max: int,
    data_domains: list,
    total_number_datapoints: int,
    evaluation_task: dict,
    eval_method: str,
    BO_params : dict,
    model_id : str,
    sampling_method="random",
    train_epochs: int = 1,
    training_batch: int = 8,
    evaluation_batch: int = 4,
    max_steps=-1,
    eval_steps=100,
    num_eval_samples=100,
    seed=42,
    what_to_optimize : str = "both",
    data_cache_dir : str = "./dataset_cache",
    num_initial_random_samples: int = 3):
    """
    Unified Bayesian Optimization loop for:
      - optimizing only data mixing ratios (optimize_data=True, optimize_lora=False)
      - optimizing only LoRA parameters (optimize_data=False, optimize_lora=True)
      - optimizing data + LoRA parameters (both True)
    
    fixed_data_ratio: list of floats summing to 1. Used when optimize_data=False
    default_lora_config: dictionary with keys ['num_layers_to_apply', 'layer_mask', 'rank', 'dropout',] 
        used when optimize_lora=False
    """
    
    # -------------------------
    # Tokenizer & model
    # -------------------------
    tokenizer, model = get_tokenizer_and_model(model_id=model_id)
    lora_max_num_layers = len(model.model.layers)
    lora_rank_max = 128  # adjust as needed
    
    # -------------------------
    # Load training datasets
    # -------------------------
    train_datasets, val_datasets = [], []
    for domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=domain, data_cache_dir=data_cache_dir)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        
    # -------------------------
    # Load evaluation datasets if our performance metric is eval_loss, else this is None
    # -------------------------
    all_sampled_evaluation_data = None
    if eval_method == "eval_loss":
        evaluation_datasets_and_weights = []
        for domain in evaluation_task.keys():
            _ , val_dataset = load_data(data_domain=domain, data_cache_dir=data_cache_dir)
            evaluation_datasets_and_weights.append((val_dataset, domain, evaluation_task[domain][0])) # 0-th index is weight
        all_sampled_evaluation_data = [] # same distribution as training data, but validation
        for eval_data, data_domain, weight in evaluation_datasets_and_weights:
            print("data domain: ", data_domain, " weight: ", weight)
            sampled_val_data = sample(eval_data, int(total_number_datapoints * weight/10), additional_info=None, method="random", data_domain=data_domain, seed=seed)
            sampled_val_data = sampled_val_data.shuffle(seed=seed).map(tokenizing_method[data_domain], fn_kwargs={"tokenizer": tokenizer,
                                                                                "add_eos_token": add_eos_token,
                                                                                "train_on_inputs": train_on_inputs,
                                                                                }, keep_in_memory=True)
            sampled_val_data = sampled_val_data.select_columns(['input_ids', 'attention_mask', 'labels'])
            all_sampled_evaluation_data.append(sampled_val_data)
        all_sampled_evaluation_data = concatenate_datasets(all_sampled_evaluation_data)

    

    # -------------------------
    # Randomly initialize first input_X
    # -------------------------
    input_X, input_X_between_0_1, fidelity = randomly_generate_data(what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max, BO_params, seed=42) # keep seed fixed for initial point
    lower_bound, upper_bound = generate_bounds(what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max, fidelity)
    bounds = torch.stack([torch.DoubleTensor(lower_bound), torch.DoubleTensor(upper_bound)])

    # -------------------------
    # Historical GP observations
    # -------------------------
    full_input, GP_input, observed_output = [], [], []
    all_influences = [None for _ in data_domains]

    # -------------------------
    # BO loop
    # -------------------------
    max_performance_so_far = float('-inf')
    results_list = []
    full_train_results_list = []
    all_fidelity_levels = []
    count_low_fidelity = 0
    count_high_fidelity = 0
    pbar = tqdm(total=BO_params["BO_iterations"])
    itr = 0
    
    # whether to apply JoBS
    # if BO_params["to_apply_joBS"]:
    #     # TBD
    #     # 1. load the performance predictor (we might want to add an arg to load a predictor trained using different number of samples)
    #     # 2. Edit the BO_run necessarily (this is to reduce the number of BO_run depending on number of data used for step 1).
    #     # 3. Create some kind of callback here (to make early loss or performance evaluations; or a sequence of it) so that it can be passed into the training function
    #     # 4. We DO NOT change the original time_callback variable to reduce the time. This is because we want to track the real performance gain from JoBS.
    #     #    Our code handles this naturally anyways.
        
    #     task = list(evaluation_task.keys())[0]
    #     predictor_path = BO_params.get("predictor_path", f"trained_predictor_fixed_20train_10val/{eval_method}_H25_50_75_100_T625_curve/{task}/performance_mlp_20samples.pth")
    #     input_dim = len(input_X) + 9    # +10 for training steps 25, 50, 75, 100 eval_loss/performance + training step 625
    #     predictor_model = MetricPredictorMLP(input_dim=input_dim)
    #     try:
    #         print("input_dim for JoBS predictor: ", input_dim)
    #         print("JoBS: Loading predictor from ", predictor_path)
    #         # Map to CPU first to avoid device conflicts
    #         predictor_model.load_state_dict(torch.load(predictor_path, map_location='cpu'))
    #         predictor_model.eval()
    #         print(f"JoBS: Predictor loaded successfully from {predictor_path}")
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"JoBS: Could not find predictor weights at {predictor_path}")
        
    #     BO_params["BO_iterations"] -= 20    # remove some iterations since we are using 20 prior observations to fit the GP initially

    #     # Fit GP with prior observations collected for the predictor
    #     print("Fitting GP with prior observations collected for JoBS performance predictor...")
    #     with open(f"results_eval_random_in_dist/['{task}']_eval_results.json", "r") as f:
    #         raw_data = json.load(f)
    #     # Process raw_data to extract GP_input and observed_output
    #     # Taking only the first 30 observations
    #     for raw_sample in raw_data[:75]:
    #         # Extract eval_loss/performance at step 625
    #         if eval_method == "performance":
    #             final_recorded_value = raw_sample['evaluations'][-1][eval_method]
    #         elif eval_method == "eval_loss":
    #             final_recorded_value = -raw_sample['evaluations'][-1][eval_method]  # need to negate for eval_loss
            
    #         # Only add if we have both input and target
    #         if 'input_X' in raw_sample and final_recorded_value is not None:
    #             previous_input_X = raw_sample['input_X']
    #             previous_input_X_between_0_1 = inverse_process_candidate(previous_input_X)
    #             print("Checking history sample input_X: ", previous_input_X)
    #             print("Checking history sample input_X_between_0_1: ", previous_input_X_between_0_1)
    #             print(f"Checking history sample {eval_method} at 625 steps: ", final_recorded_value)
    #             GP_input.append(previous_input_X_between_0_1)
    #             observed_output.append(final_recorded_value)
        
    #     # We want to use the input_X that the GP suggests using the 20 historical data, instead of randomly generating one
    #     # Fit GP
    #     print("creating a GP: MixedSingleTaskGP")
    #     discrete_dims = {
    #                 len(data_domains)+1: [0,1], # modules
    #                 len(data_domains)+2: [0,1],
    #                 len(data_domains)+3: [0,1],
    #                 len(data_domains)+4: [0,1],
    #                 len(data_domains)+5: [0,1],
    #                 len(data_domains)+9: [0,1] # reverse?
    #             }
    #     gp = MixedSingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), discrete_dims, outcome_transform=Standardize(m=1), input_transform=Normalize(d=len(input_X_between_0_1)))

    #     print("fitting GP to data since optimize_method is not random")
    #     fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

    #     # Suggest next candidate
    #     if BO_params["acq_function"] == "ucb":
    #         acq = UpperConfidenceBound(gp, beta=BO_params["ucb_beta"]/(2*(itr+1)**0.5))
    #     if BO_params["acq_function"] == "EI":
    #         acq = LogExpectedImprovement(gp, best_f=max_performance_so_far)
    #     if BO_params["optimize_method"] == "multi_fidelity":
           
    #         acq = CostScaledUCB(model=gp, beta=BO_params["ucb_beta"]/(2*(itr+1)**0.5), cost_fn=cost_fn)
    #     if BO_params["optimize_method"] == "multi_fidelity_KG":
    #         print("building KG acq function")

    #         num_fantasies = 64
    #         # base KG
    #         qKG = qKnowledgeGradient(gp, num_fantasies=num_fantasies)
        
    #         # get current best posterior mean
    #         argmax_pmean, max_pmean = optimize_acqf(
    #             acq_function=PosteriorMean(gp),
    #             bounds=bounds,
    #             q=1,
    #             num_restarts=20,
    #             raw_samples=2048,
    #         )
            
    #         acq = CostScaledKG(model=gp, cost_fn=cost_fn, num_fantasies=num_fantasies, current_max_pmean=max_pmean, sampler=qKG.sampler)

    #     # constraints on data mixing ratio
    #     A = [1.0]*len(data_domains)
    #     x = list(range(len(data_domains)))
        
    #     candidate = None
        
    #     if BO_params["optimize_method"] == "mixed" or BO_params["optimize_method"] == "multi_fidelity" or  BO_params["optimize_method"] == "multi_fidelity_KG":
            
    #         print("acq optimization method is mixed (alternating discrete and continuous)")
    #         if what_to_optimize == "data": # sum to 1
    #             candidate, acq_value = optimize_acqf_mixed_alternating(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=1024,
    #                                         equality_constraints=[(torch.tensor(x), torch.tensor(A), 1)])
    #         if what_to_optimize == "model": # no constraints
    #             candidate, acq_value = optimize_acqf_mixed_alternating(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=1024, discrete_dims=discrete_dims)
                
    #         if what_to_optimize == "both": # sum to 1 for data too
    #             t_prev = time.time()
    #             q=1
    #             if BO_params["optimize_method"] == "multi_fidelity_KG":
    #                 q = 1 + 64 # for KG, we need to add one more batch
    #                 # KG cannot used mixed
    #                 candidate, _ = optimize_acqf(
    #                     acq_function=acq,
    #                     bounds=bounds,
    #                     q=q,
    #                     num_restarts=20,
    #                     raw_samples=1024,
    #                 )
    #                 candidate = candidate[0:1, :]
    #             else: # normal mixed BO
    #                 candidate, acq_value = optimize_acqf_mixed_alternating(acq, bounds=bounds, q=q, num_restarts=20, raw_samples=1024, discrete_dims=discrete_dims,
    #                                             equality_constraints=[(torch.tensor(x), torch.tensor(A), 1)])
                
                
    #             t_now = time.time()
    #             print(f"Time taken to perform acquisition optimization: {t_now - t_prev:.4f} seconds")
                
    #             if not BO_params["optimize_method"] == "multi_fidelity_KG":
    #                 # Suppose X_sampled is a tensor of shape [n_points, d]
    #                 # Example: 5 random points within bounds
    #                 X_sampled = torch.rand(5, bounds.shape[1]) * (bounds[1] - bounds[0]) + bounds[0]
    #                 X_sampled = torch.tensor(X_sampled, dtype=torch.double)
    #                 # Evaluate acquisition
    #                 acq_values = acq(X_sampled.unsqueeze(1))

    #                 print("Sampled points and acquisition values for multi fidelity:")
    #                 for x, val in zip(X_sampled, acq_values):
    #                     print(f"X = {x.tolist()}  →  acq = {val.item()}")
    
    #         # remove last element fidelity for processing below
    #         if fidelity is not None:
    #             fidelity = round(candidate[0][-1].item())
    #             candidate = candidate[:, :-1]
            
    #     # if layers to apply loRA is 0, set last element to 1
    #     print("proposed candidate layer mask is: ", candidate[0][-9:-4])
    #     if torch.round(candidate[0][-9:-4]).sum() == 0:
    #         print("proposed candidate has all zero for layer mask, adjusting to have at least one layer to apply LoRA")
    #         # Set that slice to [0, 0, 0, 0, 1]
    #         candidate[0][-9:-4] = torch.tensor([0, 0, 0, 0, 1], dtype=candidate[0].dtype)
        
    #     input_X_between_0_1 = list(candidate[0])
    #     if what_to_optimize != "data":
    #         input_X = process_candidate(candidate[0], what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max)
    #     else:
    #         input_X = input_X_between_0_1

    #     print("input_X after fitting GP with prior data:", input_X)
    #     print("input_X_between_0_1 after fitting GP with prior data:", input_X_between_0_1)
    #     print("fidelity:", fidelity)
    
    full_evaluation_count = -1
    
    # -------------------------
    # Initial random sampling phase
    # -------------------------
    print(f"\n=== Starting Initial Random Sampling Phase ({num_initial_random_samples} samples) ===\n")
    for init_sample_idx in range(num_initial_random_samples):
        print(f"\n--- Initial Sample {init_sample_idx + 1}/{num_initial_random_samples} ---")
        
        # Generate random configuration
        input_X, input_X_between_0_1, sample_fidelity = randomly_generate_data(
            what_to_optimize, data_domains, lora_max_num_layers, 
            lora_rank_max, BO_params, seed=None
        )
        
        _, _, discrete_dims = get_lora_and_mixing_ratio(input_X, what_to_optimize, data_domains, lora_max_num_layers, default_lora_config, sample_fidelity) # just to print the generated config in a more readable way
        
        # Evaluate this configuration
        observed_perf, realized_perf, _ = evaluate_single_configuration(
            input_X=input_X,
            fidelity=sample_fidelity,
            what_to_optimize=what_to_optimize,
            data_domains=data_domains,
            lora_max_num_layers=lora_max_num_layers,
            lora_rank_max=lora_rank_max,
            default_lora_config=default_lora_config,
            model_id=model_id,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            all_sampled_evaluation_data=all_sampled_evaluation_data,
            total_number_datapoints=total_number_datapoints,
            sampling_method=sampling_method,
            train_epochs=train_epochs,
            training_batch=training_batch,
            max_steps=max_steps,
            eval_steps=eval_steps,
            evaluation_task=evaluation_task,
            eval_method=eval_method,
            evaluation_batch=evaluation_batch,
            num_eval_samples=num_eval_samples,
            time_callback=time_callback,
            seed=seed,
        )
        
        # Update max performance
        max_performance_so_far = max(max_performance_so_far, realized_perf)
        
        # Record in GP observations
        gp_input_current = input_X_between_0_1.copy() if isinstance(input_X_between_0_1, list) else input_X_between_0_1[:]
        if sample_fidelity is not None:
            gp_input_current.append(sample_fidelity)
            all_fidelity_levels.append(sample_fidelity)
        
        GP_input.append(gp_input_current)
        full_input.append(input_X)
        observed_output.append(observed_perf)
        full_train_results_list.append(realized_perf)
        results_list.append(realized_perf)
        
        print(f"Initial sample {init_sample_idx + 1} - Observed: {observed_perf}, Realized: {realized_perf}")
    
    print(f"\n=== Initial Sampling Complete. Starting BO Loop ===\n")
    
    # Now fit GP and suggest first BO candidate
    if len(GP_input) > 0:
        candidate, fidelity, gp = fit_GP_and_suggest_next_candidate(
            GP_input, observed_output, fidelity if fidelity is not None else None, 
            what_to_optimize, BO_params, max_performance_so_far, bounds, cost_fn, 
            data_domains, discrete_dims if 'discrete_dims' in locals() else {}, 
            itr, lora_max_num_layers, lora_rank_max
        )
        
        # Process candidate for next iteration
        input_X_between_0_1 = list(candidate[0])
        if what_to_optimize != "data":
            input_X = process_candidate(candidate[0], what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max)
        else:
            input_X = input_X_between_0_1
    
    while itr < BO_params["BO_iterations"]:
        print("\n\n\n")
        print("======== BO iteration: ", itr, " ==========")
        torch.manual_seed(seed)
        np.random.seed(seed)
        tokenizer, model = get_tokenizer_and_model(model_id=model_id)
        lora_config = None
        discrete_dims = None
        
        # increment iteration count and update progress bar.
        if fidelity is not None:
            if fidelity == 0.0:
                print("lower fidelity requested")
                itr += 0.5
                pbar.update(0.5)
                count_low_fidelity += 1
            else:
                print("higher fidelity requested")
                itr += 1
                pbar.update(1)
                count_high_fidelity += 1
        else:
            itr +=1
            pbar.update(1)
            count_high_fidelity += 1
        
        # print the current iteration proposed candidate in a more readable way
        print_inputs(input_X, data_domains, what_to_optimize)
        
        lora_config, mixing_ratio, discrete_dims = get_lora_and_mixing_ratio(input_X, what_to_optimize, data_domains, lora_max_num_layers, default_lora_config, fidelity)
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
          
        # generate callback if predictor is used
        callbacks_list = [time_callback]
        if eval_method == "performance":
            full_evaluation_count += 1
            class PerformanceEvalCallback(TrainerCallback):
                def __init__(self, tokenizer, eval_tasks):
                    self.tokenizer = tokenizer
                    self.target_eval_steps = {25, 50, 75, 100} 
                    self.eval_tasks = eval_tasks
                    self.step_performances = {} 

                def _evaluate_performance(self, model, trainer, current_step):
                    print(f"Running evaluation for step {current_step}...")
                    model.eval()
                    results = evaluate_tasks(list(self.eval_tasks.keys()), model, self.tokenizer, batch=8, few_shot=3, limit=num_eval_samples)
                    model.train()
                    
                    performance = None
                    for task, value in self.eval_tasks.items():
                        _, metric = value
                        performance = results["results"][task][metric]
                        break
                        
                    print(f"Evaluation performance at step {current_step}: {performance}")
                    return performance

                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step in self.target_eval_steps:
                        performance = self._evaluate_performance(kwargs.get('model'), kwargs.get('trainer'), state.global_step)
                        key_name = f"performance_step_{state.global_step}"
                        self.step_performances[key_name] = performance

            performance_eval_callback = PerformanceEvalCallback(tokenizer, evaluation_task)
            callbacks_list.append(performance_eval_callback)
        
        # Train & evaluate
        train_results = extract_data_mixture_and_train(
            model=model,
            tokenizer=tokenizer,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            data_domains=data_domains,
            evaluation_dataset=all_sampled_evaluation_data,
            mixing_ratio=mixing_ratio,
            additional_info=all_influences,
            total_number_datapoints=total_number_datapoints,
            method=sampling_method,
            train_epochs=train_epochs,
            batch_size=training_batch,
            max_step=max_steps,
            lora_config=lora_config,
            eval_steps=eval_steps,
            callback=callbacks_list,
            seed=seed
        )

        observed_performance, realized_performance = evaluate_final_performance(model, tokenizer, eval_method, fidelity, evaluation_task, train_results, evaluation_batch, num_eval_samples)
        
        # max performance
        max_performance_so_far = max(max_performance_so_far, realized_performance)
        full_train_results_list.append(realized_performance)
        results_list.append(realized_performance)
        
        print(f"current iteration observed {eval_method}: ", observed_performance)
        print(f"current iteration realized {eval_method}: ", realized_performance)
        print(f"max {eval_method} so far: ", max_performance_so_far)
        
        # Create a copy for GP input to avoid modifying original
        gp_input_current = input_X_between_0_1.copy() if isinstance(input_X_between_0_1, list) else input_X_between_0_1[:]
        
        # append fidelity if needed
        if fidelity is not None:
            gp_input_current.append(fidelity)
            all_fidelity_levels.append(fidelity)
        
        # update observations
        GP_input.append(gp_input_current)
        full_input.append(input_X)
        observed_output.append(observed_performance)
        
        # suggest next candidate
        candidate, next_fidelity, gp = fit_GP_and_suggest_next_candidate(
            GP_input, observed_output, fidelity, what_to_optimize, BO_params, 
            max_performance_so_far, bounds, cost_fn, data_domains, discrete_dims, 
            itr, lora_max_num_layers, lora_rank_max
        )
        
        # Update bounds for next iteration if fidelity changes
        if next_fidelity != fidelity:
            lower_bound, upper_bound = generate_bounds(what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max, next_fidelity)
            bounds = torch.stack([torch.DoubleTensor(lower_bound), torch.DoubleTensor(upper_bound)])
        
        # Check layer mask constraint
        if what_to_optimize != "data":    
            if torch.round(candidate[0][-9:-4]).sum() == 0:
                print("WARNING: Candidate has all zero for layer mask, adjusting...")
                candidate[0][-9:-4] = torch.tensor([0, 0, 0, 0, 1], dtype=candidate[0].dtype)
        
        input_X_between_0_1 = list(candidate[0])
        if what_to_optimize != "data":
            input_X = process_candidate(candidate[0], what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max)
        else:
            input_X = input_X_between_0_1
        fidelity = next_fidelity
        print("proposed parameters for next round by BO:", input_X)
        print("normalized proposed parameters for next round by BO:", input_X_between_0_1)
        print("fidelity for next round by BO:", next_fidelity)
    
    print("count of count_high_fidelity: ", count_high_fidelity)
    print("count of count_low_fidelity: ", count_low_fidelity)
    
    # observed_output is what the GP sees, which may include low-fidelity observations
    # full_train_results_list is the performance for full training
    if fidelity is not None:
        return GP_input, full_input, full_train_results_list, gp, all_fidelity_levels, full_train_results_list
    else: # for non-multi-fidelity, observed_output is full training performance
        return GP_input, full_input, full_train_results_list, gp, all_fidelity_levels, full_train_results_list