"""
Optimization Experiment Code for Paper [AdaPID: Adaptive Momentum Gradient Method based on PID Controller for Non-Convex Stochastic Optimization in Deep Learning]

This code implements the optimization experiments comparing various optimizers on the Rastrigin function.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from hyperopt import fmin, hp, tpe
import seaborn as sns
from optimizer.NAG import NAG
from optimizer.AdaPID import AdaPID
from optimizer.HB import HB
from optimizer.PID import PID

# Set plotting style
sns.set_style('white')


# Rastrigin function definition
def rastrigin(tensor, lib=torch):
    """Compute Rastrigin function value for given tensor"""
    x, y = tensor
    return 20 + (x ** 2 - 10 * lib.cos(x * np.pi * 2)) + (y ** 2 - 10 * lib.cos(y * np.pi * 2))


# Execute optimization steps
def execute_steps(func, initial_state, optimizer_class, optimizer_config, num_iter=100):
    """Perform optimization steps with given optimizer"""
    x = torch.tensor(initial_state, dtype=torch.float32, requires_grad=True)
    optimizer = optimizer_class([x], **optimizer_config)
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)

    # Save initial learning rate
    initial_lr = optimizer_config['lr']

    for i in range(1, num_iter + 1):
        # Compute decayed learning rate
        decayed_lr = initial_lr / i

        # Update optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = decayed_lr

        optimizer.zero_grad()
        f = func(x)
        grad = torch.autograd.grad(f, x, create_graph=True)[0]
        original_grad = grad.clone()  # Preserve unclipped gradient
        x.grad = grad
        torch.nn.utils.clip_grad_norm_(x, 1.0)

        # Handle special case for AdaPID optimizer
        if isinstance(optimizer, AdaPID):
            optimizer.step(original_grad=original_grad)  # Pass unclipped gradient
        else:
            optimizer.step()  # Normal step for other optimizers

        steps[:, i] = x.detach().numpy()
        x.grad = None  # Clear gradients to prevent memory leak

    return steps


# Rastrigin objective function for hyperparameter optimization
def objective_rastrigin(params):
    """Objective function for hyperparameter tuning"""
    lr = params['lr']
    optimizer_class = params['optimizer_class']
    minimum = (0, 0)  # Global minimum coordinates
    initial_state = (-2.0, 3.5)  # Default starting point
    optimizer_config = dict(lr=lr)
    num_iter = 100
    steps = execute_steps(rastrigin, initial_state, optimizer_class, optimizer_config, num_iter)
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


# Plot optimization trajectory on Rastrigin function
def plot_rastrigin(grad_iter, optimizer_name, lr):
    """Visualize optimization path on Rastrigin contour plot"""
    x = np.linspace(-3.5, 3.5, 250)
    y = np.linspace(-3.5, 3.5, 250)
    minimum = (0, 0)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y], lib=np)

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 35, cmap='jet')
    ax.plot(iter_x, iter_y, color='r', marker='*', linewidth=2.5, markersize=9)
    plt.plot(*minimum, 'gD')
    plt.plot(iter_x[-1], iter_y[-1])

    # Add title with optimizer name
    plt.title(f'{optimizer_name} Optimizer')

    plt.savefig(f'Figures/{optimizer_name}.eps', dpi=1200)
    plt.show()


# Main experiment execution
def execute_experiments(optimizers, objective, func, plot_func, initial_state, seed=1):
    """Run complete optimization experiments"""
    for optimizer_class, lr_low, lr_hi, *extra_args in optimizers:
        space = {
            'optimizer_class': hp.choice('optimizer_class', [optimizer_class]),
            'lr': hp.loguniform('lr', lr_low, lr_hi),
        }
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            rstate=np.random.default_rng(seed),
        )
        print(f"Best LR: {best['lr']}")

        # Prepare additional optimizer parameters if provided
        extra_kwargs = extra_args[0] if extra_args else {}
        optimizer_config = {'lr': best['lr'], **extra_kwargs}

        # Initialize optimizer with example parameters
        optimizer = optimizer_class([torch.zeros(1, requires_grad=True)], **optimizer_config)
        steps = execute_steps(func, initial_state, optimizer_class, optimizer_config, num_iter=100)
        plot_func(steps, optimizer_class.__name__, best['lr'])


if __name__ == '__main__':
    # Define list of optimizers to test
    optimizers = [
        # Baseline methods
        (PID, -8, 2),
        (HB, -8, 1),
        (NAG, -8, 2),
        (AdaPID, -8, 1.47)
    ]

    # Execute Rastrigin function optimization and visualization
    execute_experiments(optimizers, objective_rastrigin, rastrigin, plot_rastrigin, (-3.5, -3))