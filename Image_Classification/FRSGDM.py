import torch
from torch.optim import Optimizer


class FRSGDM(Optimizer):
    """
    Fletcher-Reeves Stochastic Gradient Descent with Momentum (FRSGDM) Optimizer

    Stochastic gradient descent optimizer with Fletcher-Reeves momentum coefficient adaptation.

    Parameters:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): base momentum factor (default: 0.9)
        min_momentum (float, optional): minimum momentum value (default: 0.1)
        max_momentum (float, optional): maximum momentum value (default: 0.999)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): numerical stability constant (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, min_momentum=0.1,
                 max_momentum=0.999, weight_decay=0, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if min_momentum < 0.0 or max_momentum < 0.0 or min_momentum > max_momentum:
            raise ValueError(f"Invalid momentum range: [{min_momentum}, {max_momentum}]")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, momentum=momentum, min_momentum=min_momentum,
                        max_momentum=max_momentum, weight_decay=weight_decay, eps=eps)
        super(FRSGDM, self).__init__(params, defaults)

        # Initialize state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['v'] = torch.zeros_like(p.data)  # velocity term
                state['prev_grad_norm_sq'] = None  # previous gradient squared norm

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update step counter
                state['step'] += 1

                # Calculate current gradient squared norm
                grad_norm_sq = torch.sum(grad ** 2)

                # Calculate adaptive momentum coefficient (Fletcher-Reeves method)
                if state['prev_grad_norm_sq'] is None:
                    # Use base momentum for the first step
                    adaptive_mom = group['momentum']
                else:
                    # Fletcher-Reeves ratio: β = ||g_k||^2 / ||g_{k-1}||^2
                    fr_ratio = grad_norm_sq / (state['prev_grad_norm_sq'] + group['eps'])

                    # Apply momentum range constraints
                    adaptive_mom = torch.clamp(
                        fr_ratio,
                        group['min_momentum'],
                        group['max_momentum']
                    ).item()

                # Update velocity term
                state['v'].mul_(adaptive_mom).add_(grad, alpha=-group['lr'])

                # Update parameters
                p.data.add_(state['v'])

                # Save current gradient norm for next iteration
                state['prev_grad_norm_sq'] = grad_norm_sq.item()

        return loss