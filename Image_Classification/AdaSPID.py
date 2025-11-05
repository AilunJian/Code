import torch
from torch.optim import Optimizer


class AdaSPID(Optimizer):
    """
    Adaptive Stochastic PID (AdaSPID) Optimizer

    An optimizer based on adaptive PID controller, suitable for stochastic optimization in deep learning.

    Parameters:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): initial learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0.9)
        gamma (float, optional): derivative gain (default: 0.1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): numerical stability constant (default: 1e-8)
    """

    def __init__(self, params, lr=1e-4, momentum=0.9, gamma=0.001, weight_decay=1e-4, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, momentum=momentum, gamma=gamma,
                        weight_decay=weight_decay, eps=eps)
        super(AdaSPID, self).__init__(params, defaults)

        # Initialize state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['v'] = torch.zeros_like(p.data)  # velocity term
                state['d'] = torch.zeros_like(p.data)  # derivative term
                state['prev_grad'] = torch.zeros_like(p.data)  # previous gradient
                state['prev_param'] = p.data.clone()  # previous parameter value
                state['adaptive_lr'] = group['lr']  # adaptive learning rate
                state['adaptive_mom'] = group['momentum']  # adaptive momentum

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

                # For the first step, use standard momentum update
                if state['step'] == 1:
                    state['v'] = group['momentum'] * state['v'] - group['lr'] * grad
                    p.data.add_(state['v'])
                    state['prev_grad'] = grad.clone()
                    state['prev_param'] = p.data.clone()
                else:
                    # Calculate parameter change and gradient change
                    s_k = p.data - state['prev_param']
                    y_k = grad - state['prev_grad']

                    # Calculate adaptive learning rate
                    s_norm_sq = torch.sum(s_k ** 2)
                    s_dot_y = torch.sum(s_k * y_k)

                    if s_dot_y > group['eps']:
                        # Calculate adaptive learning rate based on curvature information
                        curvature_ratio = s_norm_sq / (s_dot_y + group['eps'])

                        # Directly use curvature ratio to adjust learning rate
                        adaptive_lr = group['lr'] * curvature_ratio

                        # Set reasonable bounds
                        adaptive_lr = torch.clamp(adaptive_lr,
                                                  group['lr'] / 100,  # wider lower bound
                                                  group['lr'] * 100)  # wider upper bound
                    else:
                        adaptive_lr = torch.tensor(group['lr'], device=p.device)

                    # Calculate adaptive momentum
                    grad_norm = torch.norm(grad)
                    prev_grad_norm = torch.norm(state['prev_grad'])

                    if prev_grad_norm > group['eps']:
                        # Calculate FR ratio: current gradient norm squared / previous gradient norm squared
                        fr_ratio = (grad_norm / prev_grad_norm) ** 2

                        # Set reasonable bounds [0.5, 0.999]
                        adaptive_mom = torch.clamp(fr_ratio, 0.5, 0.999).item()
                    else:
                        adaptive_mom = group['momentum']

                    # Update velocity term
                    state['v'] = adaptive_mom * state['v'] - adaptive_lr * grad

                    # Update derivative term
                    grad_diff = grad - state['prev_grad']
                    state['d'] = adaptive_mom * state['d'] - group['gamma'] * (1 - adaptive_mom) * grad_diff

                    # Update parameters
                    p.data.add_(state['v'] + state['d'])

                    # Save current state for next iteration
                    state['prev_param'] = p.data.clone()
                    state['prev_grad'] = grad.clone()
                    state['adaptive_lr'] = adaptive_lr.item()
                    state['adaptive_mom'] = adaptive_mom

        return loss