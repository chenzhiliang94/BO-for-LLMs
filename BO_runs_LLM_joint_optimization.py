import random
from BO import joint_opt_BO_LLM_generalized
import torch

from peft import LoraConfig

from argparse import ArgumentParser
from transformers import TrainerCallback
import time
import os

parser = ArgumentParser()

# =========================
# Experiment configuration
# =========================
parser.add_argument("--model", help="model name or path")
parser.add_argument("--experiments_setting", help="either ood or in_dist")
parser.add_argument("--seed", help="seed value for single eval", type=int)
parser.add_argument("--output_dir", help="output_dir for results json file")
parser.add_argument("--save_name", help="save_name")
parser.add_argument("--data_cache_dir", help="cache dir for loading datasets", default="./dataset_cache")

# =========================
# Training configuration
# =========================
parser.add_argument("--num_data", help="total training data size", type=int, default=10000)
parser.add_argument("--epochs", help="number of training epochs", type=int, default=1)
parser.add_argument("--training_batch", help="training batch size", type=int)
parser.add_argument("--lora_rank", help="maximum LoRA rank", type=int)
parser.add_argument("--time_limit", help="training time limit")
parser.add_argument("--JoBS", help="whether to apply scaling law", type=int, default=0)

# =========================
# Evaluation configuration
# =========================
parser.add_argument("--eval_tasks", help="evaluation tasks") # comma separated list of evaluation tasks, e.g. "commonsense_qa,gsm8k"
parser.add_argument("--eval_method", help="evaluation method") # eval_loss or performance
parser.add_argument("--evaluation_batch", help="evaluation batch size", type=int) # batch size for evaluation
parser.add_argument("--evaluation_cuda", help="evaluation cuda device", default=0) # cuda to load evaluation LLM
parser.add_argument("--limit", help="number of samples for evaluation", type=int, default=100) # number of samples to evaluate downstream performance
parser.add_argument("--trials", help="number of evaluation trials", type=int, default=1) # trials to repeat entire experiment (with different random seeds)

# =========================
# Bayesian Optimization
# =========================
parser.add_argument("--run_BO_on", help="optimize over model, data, or both", default="data")
parser.add_argument("--optimize_method", help="BO optimization method")
parser.add_argument("--acq_function", help="acquisition function")
parser.add_argument("--ucb_beta", help="UCB beta parameter", type=float, default=10.0)
parser.add_argument("--iterations", help="number of BO iterations", type=int, default=50)

# random configs
parser.add_argument("--eval_random_config", help="if specified, evaluate random configs", action="store_true")
parser.add_argument("--num_random_configs", help="if generate_random_config is set, number of random configs to generate", type=int, default=100)

# specific configs, only work if run_BO_on is set to "specific"
parser.add_argument("--eval_specific_config", help="if specified, evaluate a specific config", action="store_true")
parser.add_argument("--specific_data_config", help="if specific_config is set, use this specific data config", type=str, default=None)
parser.add_argument("--specific_model_config", help="if specific_config is set, use this specific model config", type=str, default=None)

class TimerCallback(TrainerCallback):
    def __init__(self, max_duration_seconds):
        self.max_duration = int(max_duration_seconds)
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_duration:
            print(f"⏰ Max training time of {self.max_duration} seconds reached. Stopping.")
            control.should_training_stop = True
        return control


args = vars(parser.parse_args())
print("command-line args: ", args)

# verify if --output_dir exists, if not create it
if args["output_dir"] is not None:
    os.makedirs(args["output_dir"], exist_ok=True)
output_dir = str(args["output_dir"])

to_apply_joBS = bool(args["JoBS"])
eval_method = str(args["eval_method"])
model=str(args["model"])
setting=(args["experiments_setting"])
time_limit = int(args["time_limit"])
epochs=int(args["epochs"])
trials=int(args["trials"])
cuda=int(args["evaluation_cuda"])
cuda="cuda:"+str(cuda)
BO_run = int(args["iterations"])
total_data = int(args["num_data"])
tasks = str(args["eval_tasks"]).split(",")
evaluation_weights = [1/len(tasks)] * len(tasks)
lora_rank =int(args["lora_rank"])
ucb_beta = float(args["ucb_beta"])
run_BO_on = str(args["run_BO_on"])
limit = int(args["limit"])
save_name = str(args["save_name"])
data_cache_dir = str(args["data_cache_dir"])

is_random_config = bool(args["eval_random_config"])
num_random_configs = int(args["num_random_configs"])
is_specific_config = bool(args["eval_specific_config"])
specific_data_config = str(args["specific_data_config"]).split(",")
specific_model_config = str(args["specific_model_config"]).split(",")

acq_function = str(args["acq_function"])
optimize_method = str(args["optimize_method"])

training_domain_metrics = {
  "commonsense_qa": "acc,none",
  "gsm8k": "exact_match,strict-match",
  "rowan_hellaswag": "acc,none",
  "sciq": "acc_norm,none",
  "triviaqa": "exact_match,remove_whitespace",
  "truthfulqa_gen": "bleu_acc,none",
  "wikitext": "word_perplexity,none",
  "mmlu": "acc,none",
  "arc_challenge": "acc,none"
}

eval_metrics = {
  "commonsense_qa": "acc,none",
  "gsm8k": "exact_match,strict-match",
  "rowan_hellaswag": "acc,none",
  "sciq": "acc_norm,none",
  "triviaqa": "exact_match,remove_whitespace",
  "truthfulqa_gen": "bleu_acc,none",
  "wikitext": "word_perplexity,none",
  "mmlu": "acc,none",
  "arc_challenge": "acc,none"
}

# set up training data (depending if we want ood)
data_domains_initial = list(training_domain_metrics.keys())
if setting == "ood":
    data_domains =  [x for x in data_domains_initial if x not in tasks] # remove training domain that is in task
else:
    data_domains = [x for x in data_domains_initial]

# set up evaluation tasks (and weights, if we have more than one evaluation task)
evaluation_task = {}
for task, weight in zip(tasks, evaluation_weights):
    evaluation_task[task] = (float(weight), eval_metrics[task])

print("evaluation tasks and weights: ", evaluation_task)

train_epochs = int(args["epochs"])
training_batch = int(args["training_batch"])
evaluation_batch = int(args["evaluation_batch"])
evaluation_steps = 25
final_info_stored = {"command line args": args,
                    "training domain": data_domains,
                    "evaluation domain": tasks,
                    "weight": evaluation_weights} # weight in str
            
BO_params = {
    "acq_function": acq_function, # either "ucb" or "EI"
    "ucb_beta": ucb_beta,
    "optimize_method": optimize_method, # either "mixed" or "standard" or "multi_fidelity" or "multi_fidelity_KG"
    "to_apply_joBS": to_apply_joBS,
}

if model == "llama-8b":
    model_id="meta-llama/Meta-Llama-3-8B-Instruct"
elif model == "qwen-7b":
    model_id="Qwen/Qwen2.5-7B-Instruct"
elif model == "qwen-14b":
    model_id="Qwen/Qwen3-14B"
elif model == "qwen-32b":
    model_id="Qwen/Qwen3-32B"
else:
    assert False, "model not recognized"

sample_method = "random"
results = [] # GP best seen so far
full_inputs_results = [] # all inputs tried
full_train_performance_results = [] # best seen full performance (theoretical, if we performed fully)
for x in range(trials):
    
    rng = random.Random()
    seed = rng.randint(0, 1000)
    default_lora_config = LoraConfig(
            r=128,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",)
    
    print("running BO on both data and model")

    if model == "llama-8b":
        model_id="meta-llama/Meta-Llama-3-8B-Instruct"
    elif model == "qwen-7b":
        model_id="Qwen/Qwen2.5-7B-Instruct"
    elif model == "qwen-14b":
        model_id="Qwen/Qwen3-14B"
    elif model == "qwen-32b":
        model_id="Qwen/Qwen3-32B"
    else:
        assert False, "model not recognized"
    GP_input, full_inputs, observed_output, gp, all_fidelity_levels, full_train_performance = joint_opt_BO_LLM_generalized(default_lora_config=default_lora_config, 
                                                                    time_callback=TimerCallback(time_limit),
                                                                    lora_rank_max=lora_rank,
                                                                    data_domains = data_domains,
                                                                BO_run = BO_run,
                                                                total_data = total_data,
                                                                evaluation_task = evaluation_task,
                                                                eval_method=eval_method,
                                                                BO_params = BO_params,
                                                                sampling_method = sample_method, 
                                                                train_epochs=train_epochs, 
                                                                training_batch=training_batch, 
                                                                evaluation_batch=evaluation_batch,
                                                                eval_steps=evaluation_steps,
                                                                limit=limit,
                                                                seed=seed,
                                                                model_id=model_id,
                                                                what_to_optimize=run_BO_on,
                                                                data_cache_dir=data_cache_dir)

    current_max = float('-inf')  # Start with negative infinity
    max_until_now = []           # List to store max values at each step

    # Iterate through the list
    for num in observed_output:
        current_max = max(current_max, num)  # Update the current maximum
        max_until_now.append(current_max)    # Store the max up to this step

    # best performance seen by BO at every step
    print("Best at every step:", max_until_now)
    results.append(max_until_now)
    # convert to numerical
    full_inputs = [
        [x.item() if isinstance(x, torch.Tensor) else x for x in inner]
        for inner in full_inputs
    ]
    full_inputs_results.append(full_inputs)
    full_train_performance_results.append(full_train_performance)
final_info_stored[sample_method + "_best_seen"] = results
final_info_stored[sample_method + "_full_inputs"] = full_inputs_results
final_info_stored[sample_method + "_full_train_performance"] = full_train_performance
    
    
import json
import os

print("final results: ", final_info_stored)
# Combine the info you want to save

output_data = {
    "final_info_stored": final_info_stored,
    "full_training_run_performance": full_train_performance,
    "BO_params": BO_params,
    "fidelity_levels": all_fidelity_levels
}

# Define a path to save the JSON file
save_path = os.path.join(output_dir, "_".join(tasks), save_name)  # You can change this to any directory you like, e.g., "/home/user/bo_results.json"

# Optionally create the directory if it doesn't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Write to JSON
with open(save_path, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Results saved to {save_path}")







