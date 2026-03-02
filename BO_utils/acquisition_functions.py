from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound, PosteriorMean
import torch    

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
    