import os
import torch
import random
import numpy as np
from typing import List
from peft import LoraConfig

def print_non_lora_params(model: torch.nn.Module, title: str = "") -> None:
    """Print the first 5 non-LoRA parameters of the model for debugging.

    Iterates over all named parameters and prints the first 5 values
    of up to 5 parameters that are not LoRA-specific.

    Args:
        model: The model (possibly with LoRA adapters) to inspect.
        title: Optional section title printed as a header.
    """
    print(f"\n===== {title} =====")
    count = 0
    for name, param in model.named_parameters():
        # Exclude LoRA parameters
        if "lora_" not in name:
            print(name, param.data.view(-1)[:5])
            count += 1
            if count >= 5:
                break
            
def print_lora_params(model: torch.nn.Module, title: str = "") -> None:
    """Print the first 5 trainable LoRA parameters of the model for debugging.

    Iterates over all named parameters and prints the first 5 values
    of up to 5 LoRA parameters that require gradients.

    Args:
        model: The model (with LoRA adapters) to inspect.
        title: Optional section title printed as a header.
    """
    print(f"\n===== {title} =====")
    count = 0
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            print(name, param.data.view(-1)[:5])
            count += 1
            if count >= 5:
                break

def sample_random_mask(n: int = 5) -> List[int]:
    """Generate a random binary mask of length n, ensuring at least one 1.

    Each element is independently sampled as 0 or 1. If all elements
    happen to be 0, the last element is forced to 1 so that at least
    one module is always selected.

    Args:
        n: Length of the mask to generate.

    Returns:
        A list of n integers, each 0 or 1, with at least one 1.
    """
    # Sample random 0/1 mask
    mask = [random.choice([0, 1]) for _ in range(n)]
    
    # If all zeros, force last element to 1
    if sum(mask) == 0:
        mask[-1] = 1
    
    return mask

def randomly_generate_data(
    what_to_optimize: str,
    data_domains: List[str],
    lora_max_num_layers: int,
    lora_rank_max: int,
    BO_params: dict,
    seed: int | None = None
) -> tuple[List[float], List[float], int | None]:
    """Generate a BO candidate configuration using a scrambled Sobol sequence.

    Uses a low-discrepancy Sobol quasi-random sequence instead of
    pseudo-random sampling, giving better space-filling coverage of the
    search space — especially beneficial for the initial design phase
    of Bayesian Optimisation.

    Depending on ``what_to_optimize``, generates either data mixing ratios
    (mapped from Sobol via inverse-exponential → normalised, equivalent to
    Dirichlet(1,…,1)), LoRA hyper-parameters, or both.  Also picks a
    fidelity level when the optimisation method requires it.

    Args:
        what_to_optimize: One of ``"data"``, ``"model"``, or ``"both"``.
        data_domains: Names of the data domains (determines Dirichlet dimension).
        lora_max_num_layers: Maximum number of model layers available for LoRA.
        lora_rank_max: Maximum LoRA rank.
        BO_params: Bayesian Optimisation config dict (must contain ``"optimize_method"``).
        seed: Scramble seed for the Sobol engine.  Different seeds produce
            different scrambled sequences.  ``None`` uses a random scramble.

    Returns:
        input_X: Actual parameter values.
        input_X_between_0_1: Parameters normalised to [0, 1].
        fidelity: Randomly chosen fidelity level (0 or 1), or ``None`` for single-fidelity.
    """
    from torch.quasirandom import SobolEngine

    # ── Determine total Sobol dimension ──────────────────────────────
    k = len(data_domains)
    model_dims = 10  # num_layers(1) + mask(5) + rank(1) + dropout(1) + alpha(1) + reverse(1)

    if what_to_optimize == "data":
        dim = k
    elif what_to_optimize == "model":
        dim = model_dims
    else:  # "both"
        dim = k + model_dims

    needs_fidelity = BO_params["optimize_method"] in (
        "multi_fidelity", "multi_fidelity_KG"
    )
    if needs_fidelity:
        dim += 1

    # ── Draw one Sobol point in [0, 1)^dim ───────────────────────────
    # Use os.urandom instead of random.randint so that the seed is not
    # affected by random.seed() calls from training libraries (e.g.
    # transformers.set_seed), which would reset Python's random state and
    # cause identical scramble seeds on every subsequent call.
    scramble_seed = seed if seed is not None else int.from_bytes(os.urandom(4), 'big') % (2**31)
    print("scramble seed: ", scramble_seed)
    engine = SobolEngine(dimension=dim, scramble=True, seed=scramble_seed)
    raw = engine.draw(1).squeeze(0).numpy()  # shape (dim,)

    input_X: List[float] = []
    input_X_between_0_1: List[float] = []
    idx = 0  # running index into `raw`

    # ── Data mixing ratios (Dirichlet(1,…,1) via inverse Exp CDF) ────
    if what_to_optimize in ("data", "both"):
        u = raw[idx : idx + k]
        # Inverse CDF of Exponential(1): -log(1 - u), then normalise
        exp_vals = -np.log(1.0 - u + 1e-10)
        mix = (exp_vals / exp_vals.sum()).tolist()
        input_X.extend(mix)
        input_X_between_0_1.extend(mix)  # already in [0, 1]
        idx += k

    # ── Model hyper-parameters ───────────────────────────────────────
    if what_to_optimize in ("model", "both"):
        # num_layers_to_apply: [0,1) → {1, …, lora_max_num_layers}
        num_layers = min(lora_max_num_layers, int(raw[idx] * lora_max_num_layers) + 1)
        input_X.append(num_layers)
        input_X_between_0_1.append(num_layers / lora_max_num_layers)
        idx += 1

        # module mask: 5 binary values, threshold at 0.5
        mask = [1 if raw[idx + i] >= 0.5 else 0 for i in range(5)]
        if sum(mask) == 0:
            mask[-1] = 1  # guarantee at least one module
        input_X.extend(mask)
        input_X_between_0_1.extend(mask)
        idx += 5

        # rank: [0,1) → {1, …, lora_rank_max}
        rank = min(lora_rank_max, int(raw[idx] * lora_rank_max) + 1)
        input_X.append(rank)
        input_X_between_0_1.append(rank / lora_rank_max)
        idx += 1

        # dropout: [0,1) → [0, 0.1]
        dropout = round(float(raw[idx]) * 0.1, 4)
        input_X.append(dropout)
        input_X_between_0_1.append(dropout / 0.1)
        idx += 1

        # alpha: [0,1) → {1, …, 48}
        alpha = min(48, int(raw[idx] * 48) + 1)
        input_X.append(alpha)
        input_X_between_0_1.append(alpha / 48)
        idx += 1

        # reverse: binary, threshold at 0.5
        reverse = 1 if raw[idx] >= 0.5 else 0
        input_X.append(reverse)
        input_X_between_0_1.append(reverse)
        idx += 1

    # ── Fidelity ─────────────────────────────────────────────────────
    fidelity = None
    if needs_fidelity:
        fidelity = 1 if raw[idx] >= 0.5 else 0
        print("fidelity is required. Sobol-generated fidelity:", fidelity)
    else:
        print("fidelity is not required")

    return input_X, input_X_between_0_1, fidelity

def print_inputs(
    input_X: List[float],
    data_domains: List[str],
    run_bo_on: str = "both"
) -> None:
    """Pretty-print a candidate configuration for logging.

    Displays data domain mixing ratios and, when model parameters are
    being optimised, the LoRA hyper-parameters (rank, dropout, alpha,
    module mask, number of layers, and reverse flag).

    Args:
        input_X: Candidate parameter vector (actual values, not normalised).
        data_domains: Names of the data domains.
        run_bo_on: One of ``"data"``, ``"model"``, or ``"both"``.  When
            ``"data"``, only mixing ratios are printed.
    """
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
    
def arrange_lora_config(
    lora_r: float,
    lora_dropout: float,
    num_layers_to_apply: int,
    five_dim_vector: List[float],
    lora_alpha: float,
    lora_reverse: bool,
    max_num_layers: int
) -> LoraConfig | None:
    """Build a LoraConfig targeting specific model layers and modules.

    Constructs a ``LoraConfig`` by selecting which transformer layers and
    which projection modules (q/v/k/up/down/gate) to apply LoRA to, based
    on the binary ``five_dim_vector`` mask and the number of layers.

    Args:
        lora_r: LoRA rank (will be cast to int).
        lora_dropout: Dropout probability for LoRA layers.
        num_layers_to_apply: How many consecutive layers receive LoRA adapters.
        five_dim_vector: Binary mask of length 5 selecting among
            ``[q_proj, v_proj, up_proj, down_proj, gate_proj]``.
        lora_alpha: LoRA scaling alpha.
        lora_reverse: If ``True``, apply LoRA to front layers; otherwise rear layers.
        max_num_layers: Total number of layers in the base model.

    Returns:
        A ``LoraConfig`` ready to be applied to a model, or ``None`` if
        ``five_dim_vector`` is all zeros (no modules selected).
    """
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

def process_candidate(
    candidate: torch.Tensor,
    what_to_optimize: str,
    data_domains: List[str],
    lora_max_num_layers: int,
    lora_rank_max: int
) -> List[float]:
    """Map a normalised [0, 1] candidate back to actual parameter values.

    Reverses the normalisation applied by the GP / acquisition optimiser so
    that parameters can be used directly for model configuration.  Does **not**
    include the fidelity dimension.

    Args:
        candidate: 1-D tensor of normalised parameter values.
        what_to_optimize: One of ``"data"``, ``"model"``, or ``"both"``.
        data_domains: Data domain names (determines the number of mixing-ratio dims).
        lora_max_num_layers: Scaling factor for the number-of-layers parameter.
        lora_rank_max: Scaling factor for the LoRA rank parameter.

    Returns:
        A list of actual (denormalised) parameter values.
    """
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

def inverse_process_candidate(
    processed_candidate: List[float],
    what_to_optimize: str,
    data_domains: List[str],
    lora_max_num_layers: int,
    lora_rank_max: int
) -> List[float]:
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

def generate_bounds(
    what_to_optimize: str,
    data_domains: List[str],
    lora_max_num_layers: int,
    lora_rank_max: int,
    fidelity: int | None
) -> tuple[List[float], List[float]]:
    """Generate lower and upper bounds for the normalised search space.

    Builds bound vectors whose length matches the candidate dimensionality.
    Data mixing-ratio dimensions are bounded in [0, 1].  Model hyper-parameter
    dimensions use small positive lower bounds to avoid degenerate configs.
    An extra [0, 1] dimension is appended when fidelity is active.

    Args:
        what_to_optimize: One of ``"data"``, ``"model"``, or ``"both"``.
        data_domains: Data domain names (determines number of mixing-ratio dims).
        lora_max_num_layers: Max layers, used to compute lower bound offset.
        lora_rank_max: Max rank, used to compute lower bound offset.
        fidelity: Current fidelity value; if not ``None``, a fidelity dim is appended.

    Returns:
        A ``(lower_bound, upper_bound)`` tuple of float lists.
    """
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
    # add to bounds for fidelity if needed
    if fidelity is not None:
        lower_bound.append(0.0)
        upper_bound.append(1.0)
    return lower_bound, upper_bound
