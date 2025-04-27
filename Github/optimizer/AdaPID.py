import torch
from torch.optim.optimizer import Optimizer, required


class AdaPID(Optimizer):
    def __init__(self, params, lr=required, beta=0.9, weight_decay=0):

        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        self.initial_beta = beta  # Save initial beta value
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super(AdaPID, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, original_grad=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): closure that reevaluates model and returns loss
            original_grad (Tensor, optional): original unclipped gradient for beta update
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            K_d = lr / 10  # Derivative gain coefficient
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Apply L2 regularization
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                param_state = self.state[p]

                # Initialize state buffers
                if 'momentum_buffer' not in param_state:
                    m_k = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    v_k = param_state['grad_diff_buffer'] = torch.zeros_like(grad)
                    prev_grad = param_state['prev_grad'] = torch.clone(grad).detach()
                    prev_original_grad = param_state['prev_original_grad'] = torch.clone(
                        original_grad).detach() if original_grad is not None else torch.clone(grad).detach()
                    param_state['is_first_step'] = True  # Flag for initial step
                else:
                    m_k = param_state['momentum_buffer']
                    v_k = param_state['grad_diff_buffer']
                    prev_grad = param_state['prev_grad']
                    prev_original_grad = param_state['prev_original_grad']
                    param_state.get('is_first_step', False)

                # Update beta using gradient norms
                if original_grad is not None and not param_state.get('is_first_step', False):
                    norm_original_grad = torch.norm(original_grad) ** 2
                    norm_prev_original_grad = torch.norm(prev_original_grad) ** 2
                    beta = norm_original_grad / (norm_prev_original_grad + 1e-8)
                    beta = max(0.7, min(beta, 0.999))  # Clip beta values
                else:
                    beta = self.initial_beta  # Use initial beta on first step
                    param_state['is_first_step'] = False

                # Momentum update
                m_k.mul_(beta).add_(grad)

                # Gradient difference update
                grad_diff = grad - prev_grad
                v_k.mul_(beta).add_(-grad_diff, alpha=(1 - beta))

                # Parameter update
                p.add_(m_k, alpha=-lr).add_(v_k, alpha=K_d)

                # Save current gradients
                param_state['prev_grad'] = grad.clone()
                param_state['prev_original_grad'] = original_grad.clone() if original_grad is not None else grad.clone()

        return loss