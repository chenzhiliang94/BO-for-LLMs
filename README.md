# Introduction
This repository contains various toolkits to run Bayesian Optimization for various LLM applications. Bayesian Optimization (BO) is an iterative algorithm used to optimize black-box functions. These black functions have no analytical form and evaluating each function value is expensive. BO has become a popular method to optimize various training ingredients of LLMs.

![eval_gsm8k_loss_but_expected](https://github.com/user-attachments/assets/55583fab-5a95-4eea-aaac-bb8ddedd9a1c)


# Setting up
0. `git clone git@github.com:chenzhiliang94/BO-for-LLMs.git`
1. Set up a virtual env of your choice (e.g., conda env)
2. `pip3 install -r requirements.txt`
3. `mkdir results printouts dataset_cache` (make sure your create these folders in the current repo)

# Install the evaluation toolkit in a separate directory   
4. `git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness` (Clone this somewhere else)
5. `cd lm-evaluation-harness`
6. `pip install -e .` (in the current env)

# To run:
1. Simply run `./BO_run_optimization.sh`. (See section later for argument arrangement in the script).

# Optimizing Training Data Mixtures
Given a downstream target task, what is the optimal fine-tuning data mixture to maximize the LLM performance on the downstream task?
We consider $n$ data domains (e.g., ArXiv, Wikipedia, math, etc) and hence, the black-box function is $\Delta^{n-1} \mapsto \mathbb{R}$; i.e., a mapping from a probability simplex of mixing ratio to the downstream performance value (a numerical value e.g., accuracy on the task). We want to find the optimal mixing data that maximizes downstream performance.

Our ICLR 2026 paper https://arxiv.org/abs/2502.00270 uses BO to optimize data mixtures.

```
@misc{chen2025duetoptimizingtrainingdata,
      title={DUET: Optimizing Training Data Mixtures via Feedback from Unseen Evaluation Tasks}, 
      author={Zhiliang Chen and Gregory Kang Ruey Lau and Chuan-Sheng Foo and Bryan Kian Hsiang Low},
      year={2025},
      eprint={2502.00270},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.00270}, 
}
```

Given a downstream task of `gsm8k` and **the performance as the task loss**, our algorithm eventually infers that it's best to use 100% of gsm8k data to get the best downstream task loss (without knowing the downstreeam task is gsm8k).
![eval_gsm8k_loss_but_expected](https://github.com/user-attachments/assets/55583fab-5a95-4eea-aaac-bb8ddedd9a1c)

An interesting phenomenon happens if we instead consider **the performance as the task performance (e.g., accuracy over gsm8k questions)**. Somehow the algorithm eventually decides that using a big portion of TriviaQA and CommonsenseQA dataset leads to the best downstream gsm8k performance!
![eval_gsm8k_performance_but_weird](https://github.com/user-attachments/assets/ee27b3d7-c9be-4f63-ad83-ba91932a208f)



# Optimizing Both Data and Fine-tuning configurations for LLMs
In another paper, we also jointly optimized the fine-tuning model configuration and data mixture for LLMs. The setup is similar to that of the previous section. To run it, simply run `./BO_data_and_optimization.sh`.

```
@misc{chen2026chickeneggdilemmacooptimizing,
      title={The Chicken and Egg Dilemma: Co-optimizing Data and Model Configurations for LLMs}, 
      author={Zhiliang Chen and Alfred Wei Lun Leong and Shao Yong Ong and Apivich Hemachandra and Gregory Kang Ruey Lau and Chuan-Sheng Foo and Zhengyuan Liu and Nancy F. Chen and Bryan Kian Hsiang Low},
      year={2026},
      eprint={2602.08351},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.08351}, 
}
```

# More fun stuff

Our code is relatively abstract. Hence, it is very easy to adjust evaluation task, optimization method, and more. You can try it out yourself to optimize the training ingredients for different tasks!

The key is simply to modify the sweep variables in `BO_run_optimization.sh`:
```
# ----------------------- #
# Example of Sweep variables (feel free to change it in `./BO_run_optimization.sh`)
# ----------------------- #

OPT_METHODS=("random" "multi_fidelity" "multi_fidelity_KG" "mixed")
ACQ_FUNCS=("ucb" "EI")
EVAL_METHODS=("eval_loss" "performance")
RUN_BO_ON_OPTIONS=("model" "data" "both")
MODELS=("llama-8b" "qwen-7b")
TRAINING_TASKS_OPTIONS=("ALL" "triviaqa,truthfulqa_gen,gsm8k,commonsense_qa,arc_challenge")

# evaluation tasks
TASKS=("arc_challenge" "triviaqa,truthfulqa_gen")
```

The sweep variables control what experiments we run.
- `OPT_METHODS` indicates what BO algorithm will be used to optimize the variables.
    - `random`: choose random inputs.
    - `multi_fidelity`: choose multi-fidelity BO.
    - `multi_fidelity_KG`: choose multi-fidelity BO with the knowledge gradient acq. function.
    - `mixed`: this is simply vanilla BO (the word mixed is used because it is a mixed optimization problem of discrete and continuous variable)
- `ACQ_FUNCS` indicates the acquisition function used.
    - `ucb`: UCB
    - `EI`: Expected Improvement
- `EVAL_METHODS` indicates the downstream performance that we are optimizing for.
    - `eval_loss`: for the downstream evaluation task, we sample a set of validation dataset and evaluate the LLM loss over it.
    - `performance`: we evaluate the downstream evaluation task performance with `lm-evaluation-harness` toolkit that we asked you to install earlier.
- `RUN_BO_ON_OPTIONS` indicates the training ingredients that should be optimized.
    - `model`: optimize LoRA configurations
    - `data`: optimize training data mixture (As we did in our ICLR 2026 paper)
    - `both`: optimize both
- `MODELS` indicates the model that we are optimizing. We support `llama-8b`, `qwen-7b`, `qwen-14b`, qwen-14b right now.
- `TRAINING_TASKS_OPTIONS` is the most fun part. You can indicate what training data domain to look at.
    - `ALL` means we consider the entire training data domains (10 for now), and optimize it to find the best mixture
    - `triviaqa,truthfulqa_gen,gsm8k,commonsense_qa,arc_challenge` means we consider 5 training data domains, and so we are optimizing a 5 dimensional data mixture.
- `TASKS` indicates what downstream evaluation task is evaluated. For example, `"arc_challenge"` indicates the `arc_challenge` task alone is evaluated. You can indicate even a mixture of downstream tasks --- e.g., `"triviaqa,truthfulqa_gen"` indicates two tasks will be evaluated with a 50:50 weightage (with eval loss or performance, as indicated by `EVAL_METHODS` argument).
- Hence, `TRAINING_TASKS_OPTIONS` and `TASKS` allow you to play with different combination of training domains and target evaluation task.

**Each variable takes in a space-separated tuple, and we sweep through the variables (in a for loop) and run experiments for every combination of variables.**
So, you can add in a bunch of options separated by a white space (see the example segment of variables above) and the script will repeat the optimization for every single combination of variable.


