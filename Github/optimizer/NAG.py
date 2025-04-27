import torch
from torch.optim.optimizer import Optimizer, required

class NAG(Optimizer):
    def __init__(self, params, lr=required, beta=0.9, weight_decay=0, lookahead_steps=5):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0 or beta > 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if lookahead_steps < 1:
            raise ValueError(f"Invalid lookahead steps value: {lookahead_steps}")

        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, lookahead_steps=lookahead_steps)
        super(NAG, self).__init__(params, defaults)

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
            lr = group['lr']
            beta = group['beta']
            weight_decay = group['weight_decay']
            lookahead_steps = group['lookahead_steps']

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
                else:
                    m_k = param_state['momentum_buffer']

                lookahead_grad = grad.clone()
                lookahead_param = p.clone()
                for _ in range(lookahead_steps):
                    m_k.mul_(beta).add_(lookahead_grad)
                    lookahead_param.add_(m_k, alpha=-lr)
                    lookahead_param.requires_grad_(True)
                    if closure is not None:
                        with torch.enable_grad():
                            closure_loss = closure()
                        lookahead_grad = torch.autograd.grad(closure_loss, lookahead_param)[0]
                    lookahead_param.requires_grad_(False)

                m_k.mul_(beta).add_(grad)
                p.add_(m_k, alpha=-lr)

                # Save the current gradient for the next iteration
                param_state['prev_grad'] = grad.clone()

        return loss