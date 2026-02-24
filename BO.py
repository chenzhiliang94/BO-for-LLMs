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
import random
import os
from tqdm import tqdm
from typing import List
import json

from datasets import concatenate_datasets
from LLM.llm import sample, tokenizing_method, train_on_inputs, add_eos_token, evaluate_tasks, get_tokenizer_and_model, load_data, extract_data_mixture_and_train
from LLM.tokenize_util import tokenizing_method
from train_predictor_variable import MetricPredictorMLP

from transformers import TrainerCallback

os.environ["HF_ALLOW_CODE_EXECUTION"] = "1"

from peft import (
    LoraConfig,
    get_peft_model,
)

lora_alpha = 16
lora_dropout= 0.05
lora_r=16
scaling_weight=1.08
kernel_w=1.05
lora_target_modules = [
    "q_proj",
    "v_proj",
]
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

def print_non_lora_params(model, title=""):
    print(f"\n===== {title} =====")
    count = 0
    for name, param in model.named_parameters():
        # Exclude LoRA parameters
        if "lora_" not in name:
            print(name, param.data.view(-1)[:5])
            count += 1
            if count >= 5:
                break
            
def print_lora_params(model, title=""):
    print(f"\n===== {title} =====")
    count = 0
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            print(name, param.data.view(-1)[:5])
            count += 1
            if count >= 5:
                break
        
# Convert all elements to float or int for JSON serialization
def to_serializable(val):
    if isinstance(val, torch.Tensor):
        val = val.item()
    return float(val) if isinstance(val, float) or isinstance(val, np.floating) else int(val) if isinstance(val, int) or isinstance(val, np.integer) else val

# initialize a list of size n, with random 0/1 values
def sample_random_mask(n=5):
    # Sample random 0/1 mask
    mask = [random.choice([0, 1]) for _ in range(n)]
    
    # If all zeros, force last element to 1
    if sum(mask) == 0:
        mask[-1] = 1
    
    return mask

def randomly_generate_data(what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max, BO_params, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed()
        np.random.seed()
    if what_to_optimize == "data":
        k = len(data_domains)
        input_X = np.random.dirichlet([1.0] * k).tolist()
        input_X_between_0_1 = input_X[:] # already between 0 and 1
        
    elif what_to_optimize == "model":
        # num_layers_to_apply
        num_layers = random.randint(1, lora_max_num_layers)
        input_X = [num_layers]
        input_X_between_0_1 = [num_layers / lora_max_num_layers]

        # random mask of 0/1 for 5 layers
        mask = sample_random_mask(5)
        input_X += mask
        input_X_between_0_1 += mask

        # rank
        rank = random.randint(1, lora_rank_max)
        input_X.append(rank)
        input_X_between_0_1.append(rank / lora_rank_max)

        # dropout
        dropout = random.uniform(0.0, 0.1)   # adjust domain if needed
        input_X.append(dropout)
        input_X_between_0_1.append(dropout / 0.1)

        # alpha
        alpha = random.randint(1, 48)
        input_X.append(alpha)
        input_X_between_0_1.append(alpha / 48)

        # reverse
        reverse = random.choice([0, 1])
        input_X.append(reverse)
        input_X_between_0_1.append(reverse)
    else: # optimize both
        
        # data mixture
        k = len(data_domains)
        input_X = np.random.dirichlet([1.0] * k).tolist()
        input_X_between_0_1 = input_X[:] # already between 0 and 1
        
        # num_layers_to_apply
        num_layers = random.randint(1, lora_max_num_layers)
        input_X.append(num_layers)
        input_X_between_0_1.append(num_layers / lora_max_num_layers)

        # random mask of 0/1 for 5 layers
        mask = sample_random_mask(5)
        input_X += mask
        input_X_between_0_1 += mask

        # rank
        rank = random.randint(1, lora_rank_max)
        input_X.append(rank)
        input_X_between_0_1.append(rank / lora_rank_max)

        # dropout
        dropout = random.uniform(0.0, 0.1)
        input_X.append(dropout)
        input_X_between_0_1.append(dropout / 0.1)

        # alpha
        alpha = random.randint(1, 48)
        input_X.append(alpha)
        input_X_between_0_1.append(alpha / 48)

        # reverse
        reverse = random.choice([0, 1])
        input_X.append(reverse)
        input_X_between_0_1.append(reverse)
        
    print("We are at initial step. BO_params[\"optimize_method\"]:", BO_params["optimize_method"])
    fidelity = None
    if BO_params["optimize_method"] == "multi_fidelity" or BO_params["optimize_method"] == "multi_fidelity_KG":
        fidelity = random.choice([0, 1])
        print("fidelity is required. Randomly generating fidelity: ", fidelity)
    else:
        print("fidelity is not required")
    return input_X, input_X_between_0_1, fidelity

# cost per fidelity
def cost_fn(X):
    fidelity = X[..., -1]          # assuming last column is fidelity
    return 1 + fidelity         # Example (you choose your own)

class CostScaledLogEI(AcquisitionFunction):
    def __init__(self, model, best_f, cost_fn, current_itr):
        super().__init__(model)
        self.log_ei = LogExpectedImprovement(model=model, best_f=best_f)
        self.cost_fn = cost_fn  # cost_fn: X -> cost
        self.current_itr = current_itr
        
    def forward(self, X):
        logei_val = self.log_ei(X)
        cost = self.cost_fn(X)

        return - logei_val / cost.squeeze(-1) # divide acquisition value with the cost to get improvement per cost

class CostScaledUCB(AcquisitionFunction):
    def __init__(self, model, beta, cost_fn):
        super().__init__(model)
        self.ucb = UpperConfidenceBound(model=model, beta=beta)
        self.cost_fn = cost_fn  # X -> cost

    def forward(self, X):
        """
        X: batch_shape x q x d
        """
        ucb_val = self.ucb(X)              # shape: batch_shape
        cost = self.cost_fn(X).squeeze(-1) # shape: batch_shape

        return ucb_val / cost

class CostScaledKG(AcquisitionFunction):
    def __init__(
        self,
        model,
        cost_fn,
        num_fantasies,
        current_max_pmean,
        sampler,
    ):
        super().__init__(model)

        self.num_fantasies = num_fantasies

        self.kg = qKnowledgeGradient(
            model=model,
            num_fantasies=num_fantasies,
            current_value=current_max_pmean,
            sampler=sampler,
        )

        self.cost_fn = cost_fn

    def forward(self, X):
        """
        X: batch_shape x (1 + num_fantasies) x d
        """

        # KG value: scalar per batch
        kg_val = self.kg(X)  # shape: batch_shape

        # Cost of the REAL point only
        real_X = X[..., :1, :]                     # batch x 1 x d
        cost = self.cost_fn(real_X)                # batch x 1 or batch
        cost = cost.squeeze(-1).squeeze(-1)        # batch

        return kg_val / cost
    
def print_inputs(
    input_X,
    data_domains: List[str],
    run_bo_on="both"
):
    idx = len(data_domains)
    mixing_ratio = input_X[:idx]
    # check lengths
    assert len(data_domains) == len(mixing_ratio), "length of data domains and mixing ratio should be the same"

    print("=== Candidate ===")
    
    print("\nData Domains and Mixing Ratios:")
    for domain, ratio in zip(data_domains, mixing_ratio):
        if isinstance(ratio, torch.Tensor):
            ratio = ratio.item()
        print(f"  {domain}: {round(ratio,3)}")
    
    if run_bo_on != "data":
        lora_r=input_X[-4]
        lora_dropout=input_X[-3]
        num_layers_to_apply=input_X[idx]
        five_dim_vector=input_X[idx+1:idx+1+5]
        lora_alpha = input_X[-2]
        lora_reverse = input_X[-1]
        
        print("number of layers to apply lora: ", num_layers_to_apply)
        print("which modules to apply lora (q_proj, v_proj, up_proj, down_proj, gate_proj): ", five_dim_vector) 
        print("lora rank: ", lora_r)   
        print("lora dropout: ", lora_dropout)
        print("lora alpha: ", lora_alpha)
        print("lora reverse (apply to rear layers if False, else front layers): ", lora_reverse)
    
    print("=========================\n")
    
def arrange_lora_config(lora_r, lora_dropout, num_layers_to_apply, five_dim_vector, lora_alpha, lora_reverse : bool, max_num_layers):
    '''
    lora_r: float
    lora_dropout = float 
    num_layers_to_apply = int
    five_dim_vector = List[float]. Five dimension
    lora_alpha = alpha
    lora_reverse = whether to reverse apply the lora to front layers or rear layers
    max_num_layers = int. Maximum number of layers in the model
    '''
    lora_r = int(lora_r)
    num_layers_to_apply = int(num_layers_to_apply)
    five_dim_vector = [int(x) for x in five_dim_vector] # convert to int
    if sum(five_dim_vector) == 0:
        return None
    print("creating lora config with parameters, to be trained.")

    # only .mlp layers have up, down, gate proj
    # only .self_attn layers have q, v, k proj
    # ["model.layers.0.self_attn.k_proj"]
    lora_modules_all = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
    lora_module_to_tune = [mod for mod, flag in zip(lora_modules_all, five_dim_vector) if flag == 1]
    lora_specific_modules = []

    for module in lora_module_to_tune:
        if module == "q_proj" or module == "v_proj" or module == "k_proj":
            for i in range(num_layers_to_apply):
                if lora_reverse: # apply to front layers
                    lora_specific_modules.append("model.layers."+str(i)+".self_attn."+module)
                else: # apply to rear layers
                    lora_specific_modules.append("model.layers."+str(max_num_layers-1-i)+".self_attn."+module)
        else:
            for i in range(num_layers_to_apply):
                if lora_reverse:
                    lora_specific_modules.append("model.layers."+str(i)+".mlp."+module)
                else:
                    lora_specific_modules.append("model.layers."+str(max_num_layers-1-i)+".mlp."+module)
    
    config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_specific_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",)

    return config

# -------------------------
    # Candidate processing
    # input candidate is normalized
    # but we want to project them to the real values
    # note that this does not include fidelity
    # -------------------------
def process_candidate(candidate, what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max):
    processed_candidate = []
    idx = 0
    
    if what_to_optimize == "data":
        assert len(candidate) == len(data_domains), "length of candidate should be equals to data domains length, since we are only optimizing data"
        
        # only data
        for v in candidate:
            processed_candidate.append(0 if v.item()<0.01 else v)
            
        assert sum(processed_candidate) < 1, "Sum of data mixing ratios should be less than 1"
        return processed_candidate
    
    if what_to_optimize == "model":
        idx = 0
        processed_candidate.append(round(lora_max_num_layers*candidate[idx].item())) # num_layers_to_apply
        idx += 1
        processed_candidate += [round(v.item()) for v in candidate[idx:idx+5]] # layer mask
        idx += 5
        processed_candidate.append(round(lora_rank_max*candidate[idx].item())) # rank
        idx += 1
        processed_candidate.append(0.1 * candidate[idx].item()) # dropout
        idx += 1
        processed_candidate.append(48.0 * candidate[idx].item()) # alpha
        idx += 1
        processed_candidate.append(round(candidate[idx].item())) # whether to reverse apply LoRA
        
        return processed_candidate
    
    elif what_to_optimize == "both":
        assert len(candidate) == len(data_domains) + 10, "length of candidate should be equals to data domains length + 10, since we are optimizing both data and model"
        
        # data
        for v in candidate[:len(data_domains)]:
            processed_candidate.append(0 if v.item()<0.01 else v)
        
        # model
        idx = len(data_domains)
        processed_candidate.append(round(lora_max_num_layers*candidate[idx].item())) # num_layers_to_apply
        idx += 1
        processed_candidate += [round(v.item()) for v in candidate[idx:idx+5]] # layer mask
        idx += 5
        processed_candidate.append(round(lora_rank_max*candidate[idx].item())) # rank
        idx += 1
        processed_candidate.append(0.1 * candidate[idx].item()) # dropout
        idx += 1
        processed_candidate.append(48.0 * candidate[idx].item()) # alpha
        idx += 1
        processed_candidate.append(round(candidate[idx].item())) # reverse
        return processed_candidate
    
    assert False, "what_to_optimize is not properly set."

def inverse_process_candidate(processed_candidate, what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max):
    """
    Converts processed_candidate (actual values) back to candidate (normalized 0-1 values).
    
    Args:
        processed_candidate: List of actual parameter values.
        what_to_optimize: String, one of "data", "model", or "both".
        data_domains: List of data domains (needed for length calculations).
        lora_max_num_layers: Maximum number of LoRA layers (scaling factor).
        lora_rank_max: Maximum LoRA rank (scaling factor).
        
    Returns:
        Tensor of normalized candidate values (between 0 and 1).
    """
    candidate = []
    
    if what_to_optimize == "data":
        assert len(processed_candidate) == len(data_domains), "Length mismatch for 'data' optimization"
        
        # Inverse of data mixing ratios (Direct mapping)
        # Note: If the forward pass zeroed out a value < 0.01, we recover 0.0.
        candidate = [float(v) for v in processed_candidate]
        
    elif what_to_optimize == "model":
        # Expected length is 10 based on the forward function (1 num_layer + 5 mask + 1 rank + 1 dropout + 1 alpha + 1 reverse)
        assert len(processed_candidate) == 10, "Length mismatch for 'model' optimization"
        
        idx = 0
        # 1. num_layers_to_apply: reversed round(lora_max_num_layers * x)
        candidate.append(processed_candidate[idx] / lora_max_num_layers)
        idx += 1
        
        # 2. layer mask (5 values): reversed round(x)
        candidate.extend([float(x) for x in processed_candidate[idx:idx+5]])
        idx += 5
        
        # 3. rank: reversed round(lora_rank_max * x)
        candidate.append(processed_candidate[idx] / lora_rank_max)
        idx += 1
        
        # 4. dropout: reversed 0.1 * x
        candidate.append(processed_candidate[idx] / 0.1)
        idx += 1
        
        # 5. alpha: reversed 48.0 * x
        candidate.append(processed_candidate[idx] / 48.0)
        idx += 1
        
        # 6. reverse: reversed round(x)
        candidate.append(float(processed_candidate[idx]))
        
    elif what_to_optimize == "both":
        assert len(processed_candidate) == len(data_domains) + 10, "Length mismatch for 'both' optimization"
        
        # --- Data Part ---
        data_len = len(data_domains)
        # Direct mapping for data ratios
        candidate.extend([float(v) for v in processed_candidate[:data_len]])
        
        # --- Model Part ---
        p_idx = data_len 
        
        # 1. num_layers_to_apply
        candidate.append(processed_candidate[p_idx] / lora_max_num_layers)
        p_idx += 1
        
        # 2. layer mask (5 values)
        candidate.extend([float(x) for x in processed_candidate[p_idx:p_idx+5]])
        p_idx += 5
        
        # 3. rank
        candidate.append(processed_candidate[p_idx] / lora_rank_max)
        p_idx += 1
        
        # 4. dropout
        candidate.append(processed_candidate[p_idx] / 0.1)
        p_idx += 1
        
        # 5. alpha
        candidate.append(processed_candidate[p_idx] / 48.0)
        p_idx += 1
        
        # 6. reverse
        candidate.append(float(processed_candidate[p_idx]))

    return candidate

# function for generating bounds based on what_to_optimize
def generate_bounds(what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max, fidelity):
    # -------------------------
    # Bounds
    # -------------------------
    lower_bound, upper_bound = [], []
    if what_to_optimize == "data":
        lower_bound += [0]*len(data_domains)
        upper_bound += [1]*len(data_domains)
    elif what_to_optimize == "model":
        lower_bound += [1/lora_max_num_layers+0.01] + [0]*5 + [1/lora_rank_max+0.01] + [0.0] + [1/48+0.01] + [0.0]
        upper_bound += [1] + [1]*5 + [1] + [1] + [1] + [1]
    else:
        lower_bound += [0]*len(data_domains)
        upper_bound += [1]*len(data_domains)
        lower_bound += [1/lora_max_num_layers+0.01] + [0]*5 + [1/lora_rank_max+0.01] + [0.0] + [1/48+0.01] + [0.0]
        upper_bound += [1] + [1]*5 + [1] + [1] + [1] + [1]
    dim_input_variables = len(lower_bound)
    # add to bounds for fidelity if needed
    if fidelity is not None:
        lower_bound.append(0.0)
        upper_bound.append(1.0)
    return lower_bound, upper_bound

 # write a function that does the lora config arrangement below:
def get_lora_and_mixing_ratio(input_X, what_to_optimize, data_domains, lora_max_num_layers, default_lora_config, fidelity):
    
    # Build LoRA config if optimizing LoRA
    if what_to_optimize == "data":
        lora_config = default_lora_config # default LoRA
        mixing_ratio = input_X
        discrete_dims = {}
    elif what_to_optimize == "model":
        
        discrete_dims = {
                1: [0,1], # modules
                2: [0,1],
                3: [0,1],
                4: [0,1],
                5: [0,1],
                9: [0,1], # reverse?
            }
        
        mixing_ratio = [1/len(data_domains)]*len(data_domains) # default using uniform
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
        discrete_dims = {
                len(data_domains)+1: [0,1], # modules
                len(data_domains)+2: [0,1],
                len(data_domains)+3: [0,1],
                len(data_domains)+4: [0,1],
                len(data_domains)+5: [0,1],
                len(data_domains)+9: [0,1] # reverse?
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

# evaluate performance at the end of full-training.
def evaluate_final_performance(model, tokenizer, eval_method, fidelity, evaluation_task, train_results, evaluation_batch, num_eval_samples):
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

def fit_GP_and_suggest_next_candidate(GP_input, observed_output, fidelity, what_to_optimize, BO_params, max_performance_so_far, bounds, cost_fn, data_domains, discrete_dims, itr, lora_max_num_layers, lora_rank_max):
            
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
        num_fantasies = 64
        # base KG
        qKG = qKnowledgeGradient(gp, num_fantasies=num_fantasies)
    
        # get current best posterior mean
        argmax_pmean, max_pmean = optimize_acqf(
            acq_function=PosteriorMean(gp),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=2048,
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
            q = 1 + 64  # for KG, we need to add one more batch
            candidate, acq_value = optimize_acqf(
                acq_function=acq, bounds=bounds, q=q, num_restarts=20, raw_samples=1024
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
        BO_run: int,
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
        data_cache_dir : str = "./dataset_cache"):
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
    # fidelity is a running current fidelity level if multi-fidelity BO is used
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
    pbar = tqdm(total=BO_run)
    itr = 0
    
    # whether to apply JoBS
    if BO_params["to_apply_joBS"]:
        # TBD
        # 1. load the performance predictor (we might want to add an arg to load a predictor trained using different number of samples)
        # 2. Edit the BO_run necessarily (this is to reduce the number of BO_run depending on number of data used for step 1).
        # 3. Create some kind of callback here (to make early loss or performance evaluations; or a sequence of it) so that it can be passed into the training function
        # 4. We DO NOT change the original time_callback variable to reduce the time. This is because we want to track the real performance gain from JoBS.
        #    Our code handles this naturally anyways.
        
        task = list(evaluation_task.keys())[0]
        predictor_path = BO_params.get("predictor_path", f"trained_predictor_fixed_20train_10val/{eval_method}_H25_50_75_100_T625_curve/{task}/performance_mlp_20samples.pth")
        input_dim = len(input_X) + 9    # +10 for training steps 25, 50, 75, 100 eval_loss/performance + training step 625
        predictor_model = MetricPredictorMLP(input_dim=input_dim)
        try:
            print("input_dim for JoBS predictor: ", input_dim)
            print("JoBS: Loading predictor from ", predictor_path)
            # Map to CPU first to avoid device conflicts
            predictor_model.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            predictor_model.eval()
            print(f"JoBS: Predictor loaded successfully from {predictor_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"JoBS: Could not find predictor weights at {predictor_path}")
        
        BO_run -= 20    # remove some iterations since we are using 20 prior observations to fit the GP initially

        # Fit GP with prior observations collected for the predictor
        print("Fitting GP with prior observations collected for JoBS performance predictor...")
        with open(f"results_eval_random_in_dist/['{task}']_eval_results.json", "r") as f:
            raw_data = json.load(f)
        # Process raw_data to extract GP_input and observed_output
        # Taking only the first 30 observations
        for raw_sample in raw_data[:75]:
            # Extract eval_loss/performance at step 625
            if eval_method == "performance":
                final_recorded_value = raw_sample['evaluations'][-1][eval_method]
            elif eval_method == "eval_loss":
                final_recorded_value = -raw_sample['evaluations'][-1][eval_method]  # need to negate for eval_loss
            
            # Only add if we have both input and target
            if 'input_X' in raw_sample and final_recorded_value is not None:
                previous_input_X = raw_sample['input_X']
                previous_input_X_between_0_1 = inverse_process_candidate(previous_input_X)
                print("Checking history sample input_X: ", previous_input_X)
                print("Checking history sample input_X_between_0_1: ", previous_input_X_between_0_1)
                print(f"Checking history sample {eval_method} at 625 steps: ", final_recorded_value)
                GP_input.append(previous_input_X_between_0_1)
                observed_output.append(final_recorded_value)
        
        # We want to use the input_X that the GP suggests using the 20 historical data, instead of randomly generating one
        # Fit GP
        print("creating a GP: MixedSingleTaskGP")
        discrete_dims = {
                    len(data_domains)+1: [0,1], # modules
                    len(data_domains)+2: [0,1],
                    len(data_domains)+3: [0,1],
                    len(data_domains)+4: [0,1],
                    len(data_domains)+5: [0,1],
                    len(data_domains)+9: [0,1] # reverse?
                }
        gp = MixedSingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), discrete_dims, outcome_transform=Standardize(m=1), input_transform=Normalize(d=len(input_X_between_0_1)))

        print("fitting GP to data since optimize_method is not random")
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

        # Suggest next candidate
        if BO_params["acq_function"] == "ucb":
            acq = UpperConfidenceBound(gp, beta=BO_params["ucb_beta"]/(2*(itr+1)**0.5))
        if BO_params["acq_function"] == "EI":
            acq = LogExpectedImprovement(gp, best_f=max_performance_so_far)
        if BO_params["optimize_method"] == "multi_fidelity":
           
            acq = CostScaledUCB(model=gp, beta=BO_params["ucb_beta"]/(2*(itr+1)**0.5), cost_fn=cost_fn)
        if BO_params["optimize_method"] == "multi_fidelity_KG":
            print("building KG acq function")

            num_fantasies = 64
            # base KG
            qKG = qKnowledgeGradient(gp, num_fantasies=num_fantasies)
        
            # get current best posterior mean
            argmax_pmean, max_pmean = optimize_acqf(
                acq_function=PosteriorMean(gp),
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=2048,
            )
            
            acq = CostScaledKG(model=gp, cost_fn=cost_fn, num_fantasies=num_fantasies, current_max_pmean=max_pmean, sampler=qKG.sampler)

        # constraints on data mixing ratio
        A = [1.0]*len(data_domains)
        x = list(range(len(data_domains)))
        
        candidate = None
        
        if BO_params["optimize_method"] == "mixed" or BO_params["optimize_method"] == "multi_fidelity" or  BO_params["optimize_method"] == "multi_fidelity_KG":
            
            print("acq optimization method is mixed (alternating discrete and continuous)")
            if what_to_optimize == "data": # sum to 1
                candidate, acq_value = optimize_acqf_mixed_alternating(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=1024,
                                            equality_constraints=[(torch.tensor(x), torch.tensor(A), 1)])
            if what_to_optimize == "model": # no constraints
                candidate, acq_value = optimize_acqf_mixed_alternating(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=1024, discrete_dims=discrete_dims)
                
            if what_to_optimize == "both": # sum to 1 for data too
                t_prev = time.time()
                q=1
                if BO_params["optimize_method"] == "multi_fidelity_KG":
                    q = 1 + 64 # for KG, we need to add one more batch
                    # KG cannot used mixed
                    candidate, _ = optimize_acqf(
                        acq_function=acq,
                        bounds=bounds,
                        q=q,
                        num_restarts=20,
                        raw_samples=1024,
                    )
                    candidate = candidate[0:1, :]
                else: # normal mixed BO
                    candidate, acq_value = optimize_acqf_mixed_alternating(acq, bounds=bounds, q=q, num_restarts=20, raw_samples=1024, discrete_dims=discrete_dims,
                                                equality_constraints=[(torch.tensor(x), torch.tensor(A), 1)])
                
                
                t_now = time.time()
                print(f"Time taken to perform acquisition optimization: {t_now - t_prev:.4f} seconds")
                
                if not BO_params["optimize_method"] == "multi_fidelity_KG":
                    # Suppose X_sampled is a tensor of shape [n_points, d]
                    # Example: 5 random points within bounds
                    X_sampled = torch.rand(5, bounds.shape[1]) * (bounds[1] - bounds[0]) + bounds[0]
                    X_sampled = torch.tensor(X_sampled, dtype=torch.double)
                    # Evaluate acquisition
                    acq_values = acq(X_sampled.unsqueeze(1))

                    print("Sampled points and acquisition values for multi fidelity:")
                    for x, val in zip(X_sampled, acq_values):
                        print(f"X = {x.tolist()}  →  acq = {val.item()}")
    
            # remove last element fidelity for processing below
            if fidelity is not None:
                fidelity = round(candidate[0][-1].item())
                candidate = candidate[:, :-1]
            
        # if layers to apply loRA is 0, set last element to 1
        print("proposed candidate layer mask is: ", candidate[0][-9:-4])
        if torch.round(candidate[0][-9:-4]).sum() == 0:
            print("proposed candidate has all zero for layer mask, adjusting to have at least one layer to apply LoRA")
            # Set that slice to [0, 0, 0, 0, 1]
            candidate[0][-9:-4] = torch.tensor([0, 0, 0, 0, 1], dtype=candidate[0].dtype)
        
        input_X_between_0_1 = list(candidate[0])
        if what_to_optimize != "data":
            input_X = process_candidate(candidate[0], what_to_optimize, data_domains, lora_max_num_layers, lora_rank_max)
        else:
            input_X = input_X_between_0_1

        print("input_X after fitting GP with prior data:", input_X)
        print("input_X_between_0_1 after fitting GP with prior data:", input_X_between_0_1)
        print("fidelity:", fidelity)
    
    full_evaluation_count = -1
    while itr < BO_run:
        print("\n\n\n")
        print("======== BO iteration: ", itr, " ==========")
        torch.manual_seed(seed)
        np.random.seed(seed)
        tokenizer, model = get_tokenizer_and_model(model_id=model_id)
        lora_config = None
        discrete_dims = None
        
        # print the current iteration proposed candidate in a more readable way
        print_inputs(input_X, data_domains, what_to_optimize)
        lora_config, mixing_ratio, discrete_dims = get_lora_and_mixing_ratio(input_X, what_to_optimize, data_domains, lora_max_num_layers, default_lora_config, fidelity)
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
        
        # increment iteration count and update progress bar.
        # If lower fidelity is requested, we increment the iteration
        # by a smaller amount because "less optimizationbudget" is used.
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
            if BO_params["to_apply_joBS"]:
                itr += 0.1
                pbar.update(0.1)
            else:
                itr +=1
                pbar.update(1)
            count_high_fidelity += 1
        
        
        # generate callback if predictor is used to extrapolate training performance
        if eval_method == "performance":
            full_evaluation_count += 1
            class PerformanceEvalCallback(TrainerCallback):
                """
                A custom callback to run lm_eval.simple_evaluate specifically at steps 25 and 50.
                Performance metrics are logged to trainer.state.log_history for easy retrieval.
                """
                def __init__(self, tokenizer, eval_tasks):
                    self.tokenizer = tokenizer
                    # Fixed evaluation steps as requested
                    self.target_eval_steps = {25, 50, 75, 100} 
                    self.eval_tasks = eval_tasks
                    # To store results temporarily before logging
                    self.step_performances = {} 
                    print(f"Initialized PerformanceEvalCallback to evaluate only at steps {self.target_eval_steps} for tasks: {eval_tasks}")

                def _evaluate_performance(self, model, trainer, current_step):
                    # Get performance from lm_eval
                    print(f"Running evaluation for step {current_step}...")
                    model.eval()
                    results = evaluate_tasks(list(self.eval_tasks.keys()), model, self.tokenizer, batch=8, few_shot=3, limit=num_eval_samples)
                    model.train()
                    
                    # Extract the specific metric value
                    performance = None
                    for task, value in self.eval_tasks.items():
                        _, metric = value
                        performance = results["results"][task][metric]
                        break # Assuming we want the first task/metric found if multiple are passed
                        
                    print(f"Evaluation performance at step {current_step}: {performance}")
                    return performance

                def on_step_end(self, args, state, control, **kwargs):
                    """
                    Called at the end of every training step. 
                    Checks if current step is 25 or 50, evaluates, and stores the result.
                    """
                    if state.global_step in self.target_eval_steps:
                        performance = self._evaluate_performance(kwargs.get('model'), kwargs.get('trainer'), state.global_step)
                        
                        # Store with a specific key for retrieval later
                        key_name = f"performance_step_{state.global_step}"
                        self.step_performances[key_name] = performance

            performance_eval_callback = PerformanceEvalCallback(tokenizer, evaluation_task)
        
        # Train & evaluate with the given data mixture and lora configurations
        # Note that even if we are using low-fidelity observations, we still train to completion for evaluation purposes.
        train_results = extract_data_mixture_and_train(
            model=model,
            tokenizer=tokenizer,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            data_domains=data_domains,
            evaluation_dataset=all_sampled_evaluation_data, # evaluation data
            mixing_ratio=mixing_ratio,
            additional_info=all_influences,
            total_number_datapoints=total_number_datapoints,
            method=sampling_method,
            train_epochs=train_epochs,
            batch_size=training_batch,
            max_step=max_steps,
            lora_config=lora_config,
            eval_steps=eval_steps,
            callback=[time_callback, performance_eval_callback] if eval_method == "performance" else [time_callback],
            seed=seed
        )

        observed_performance, realized_performance = evaluate_final_performance(model, tokenizer, eval_method, fidelity, evaluation_task, train_results, evaluation_batch, num_eval_samples)
            
        # transform observed_performance (or eval_loss_trajectory) into a new observed_performance
        # with some variant of a predictor/extrapolator
        if BO_params["to_apply_joBS"]:
            
            print(f"Applying JoBS: Extracted {eval_method} logs for performance prediction.")
            print("Full train results log history: ", train_results)
            
            # Extract first two (training steps 25 and 50) to use for predictor
            if eval_method == "eval_loss":
                recorded_values = train_results.get(eval_method, []) # loss
                realized_performance *= (1/scaling_weight)
            else: # it's performance
                # {'performance_step_25': 0.62, 'performance_step_50': 0.59}
                # take max of realized_performance and the values in this dict
                realized_performance *= scaling_weight
                realized_performance = max(realized_performance, max(train_results["step_performances"].values()))
                # best possible performance
                recorded_values = list(train_results["step_performances"].values())
                print("recorded performance values for JoBS predictor: ", recorded_values)
                
            if len(recorded_values) >= 2:
                value_25 = recorded_values[0]
                value_50 = recorded_values[1]
                value_75 = recorded_values[2]
                value_100 = recorded_values[3]
                
                loss_input = [25, value_25, 50, value_50, 75, value_75, 100, value_100, 625]
                input = input_X + loss_input
                input_tensor = torch.tensor([input], dtype=torch.float32)
                with torch.no_grad():
                    predicted_final_value = predictor_model(input_tensor).item()
                    print("predicted final value from JoBS predictor: ", predicted_final_value)
                    if eval_method == "performance":
                        observed_performance = predicted_final_value
                    elif eval_method == "eval_loss":
                        observed_performance = -predicted_final_value
            
        # max performance
        max_performance_so_far = max(max_performance_so_far, realized_performance)
        full_train_results_list.append(realized_performance)
        results_list.append(realized_performance)
        
        print(f"current iteration observed (possibly low-fid or predicted) {eval_method}: ", observed_performance)
        print(f"current iteration best possible {eval_method} (full train run): ", realized_performance)
        print(f"max {eval_method} so far: ", max_performance_so_far)
        print("BO observations: ", results_list)
        
        # append fidelity to current inputs to GP here if fidelity is given
        if fidelity is not None:
            input_X_between_0_1.append(fidelity)
            all_fidelity_levels.append(fidelity)
        
        # update observation and current inputs to historical data
        GP_input.append(input_X_between_0_1)
        full_input.append(input_X)
        observed_output.append(observed_performance)
        
        # suggest next candidate
        candidate, next_fidelity, gp = fit_GP_and_suggest_next_candidate(GP_input, observed_output, fidelity, what_to_optimize, BO_params, max_performance_so_far, bounds, cost_fn, data_domains, discrete_dims, itr, lora_max_num_layers, lora_rank_max)
        
        # The following is no longer needed because we are now handling the inequality constraints during acquisition optimization.
        # Hence, the candidate suggested will not have all zero for layer mask. We keep this code here just in case for debugging
        # The printout SHOULD NOT appear.
        if what_to_optimize != "data":    
            # if layers to apply loRA is 0, set last element to 1
            print("proposed candidate layer mask is: ", candidate[0][-9:-4])
            if torch.round(candidate[0][-9:-4]).sum() == 0:
                print("proposed candidate has all zero for layer mask, adjusting to have at least one layer to apply LoRA")
                # Set that slice to [0, 0, 0, 0, 1]
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