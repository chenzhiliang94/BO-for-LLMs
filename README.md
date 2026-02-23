# Introduction
This repository contains various toolkits to run Bayesian Optimization for various LLM applications. Bayesian Optimization (BO) is an iterative algorithm used to optimize black-box functions. These black functions have no analytical form and evaluating each function value is expensive. BO has become a popular method to optimize various training ingredients of LLMs.


# Setting up
0. `git clone git@github.com:chenzhiliang94/BO-for-LLMs.git`
1. `pip3 install -r requirements.txt`

# Install the evaluation toolkit in a separate directory   
3. `git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness`
4. `cd lm-evaluation-harness`
5. `pip install -e .`


# Optimizing Training Data Mixtures for LLMs
Given a downstream target task, what is the optimal fine-tuning data mixture to maximize the LLM performance on the downstream task?
Given $n$ data domains (e.g., ArXiv, Wikipedia, math, etc), the black-box function is $\Delta^{n-1} \mapsto \mathbb{R}$; i.e., a mapping from a probability simplex of mixing ratio to the final performance value (a numerical value e.g., accuracy on the task).

Our ICLR 2026 paper https://arxiv.org/abs/2502.00270 uses BO to optimize such data mixtures.
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

# Optimizing Fine-tuning configurations for LLMs
Alternatively, we are also interested in finding the best fine-tuning configuration for LLMs.

# Optimizing Both

# Using Lower-fidelity Observations

