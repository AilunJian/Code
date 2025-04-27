import torch
from torch.optim.optimizer import Optimizer, required

class HB(Optimizer):
    def __init__(self, params, lr=required, beta=0.9, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0 or beta > 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super(HB, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']  # 学习率
            beta = group['beta']  # 动量衰减系数
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Apply weight decay if specified
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                param_state = self.state[p]

                # Initialize momentum (m_k) and gradient difference (v_k) buffers
                if 'momentum_buffer' not in param_state:
                    m_k = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    v_k = param_state['grad_diff_buffer'] = torch.zeros_like(grad)  # Initialize v_k as zeros
                    prev_grad = param_state['prev_grad'] = torch.clone(grad).detach()
                else:
                    m_k = param_state['momentum_buffer']
                    v_k = param_state['grad_diff_buffer']
                    prev_grad = param_state['prev_grad']

                # Update momentum m_k: m_k ← β * m_(k-1) + r_k
                m_k.mul_(beta).add_(grad)

                # Update gradient difference v_k: v_k ← β * v_(k-1) + (1 - β) * (r_k - r_(k-1))
                grad_diff = grad - prev_grad
                v_k.mul_(beta).add_(-grad_diff, alpha=(1 - beta))

                # Update the parameter: θ_(k+1) ← θ_k - lr * m_k + K_d * v_k
                p.add_(m_k, alpha=-lr)

                # Save the current gradient for the next iteration
                param_state['prev_grad'] = grad.clone()

        return loss
