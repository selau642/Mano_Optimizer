import torch
from typing import Literal

class SingleDeviceMano(torch.optim.Optimizer):
    """
    Non-distributed Manu Optimizer.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(
            lr=lr, 
            weight_decay=weight_decay, 
            momentum=momentum,
            is_even_step=True
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                """

                Manu alternating updates

                """

                M = state["momentum_buffer"]
                update = M.lerp_(p.grad, 1-state["momentum"])

                row, col = p.shape
                if state["is_even_step"]:
                    # column-wise norm
                    p_col_norm = torch.linalg.vector_norm(p, dim=0)
                    p_unit = p / p_col_norm 
                    v = M - p_unit * dimension_wise_dot(M, p_col_norm, "column")
                    v_col_norm = torch.linalg.vector_norm(v, dim=0)
                    v_unit = v / v_col_norm

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(0.2* torch.sqrt(col) * v_unit, alpha=-group["lr"])

                    self.state["is_even_step"] = False
                else:

                    # row-wise norm
                    p_row_norm = torch.linalg.vector_norm(p, dim=1)
                    p_unit = p / p_row_norm 
                    v = M - p_unit * dimension_wise_dot(M, p_row_norm, "row")
                    v_col_norm = torch.linalg.vector_norm(v, dim=1)
                    v_unit = v / v_col_norm

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(0.2* torch.sqrt(row) * v_unit, alpha=-group["lr"])

                    self.state["is_even_step"] = True

                return loss

def dimension_wise_dot(A, B, dot_type: Literal["column", "row"]):
    """
    
    Args:
        A: shape (M, N)
        B: shape (M, N)

    """
    element_wise_product = A * B 

    row, col = A.shape
    if dot_type == "column":
        col_values = torch.sum(element_wise_product, dim=0)
        return col_values.repeat((row, 1))

    elif dot_type == "row":
        row_values = torch.sum(element_wise_product, dim=1).T
        return row_values.repeat((1, col))
 
