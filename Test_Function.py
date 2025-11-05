import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==============================
# Global Plot Settings
# ==============================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 35
plt.rcParams['axes.titlesize'] = 35
plt.rcParams['axes.labelsize'] = 35
plt.rcParams['xtick.labelsize'] = 35
plt.rcParams['ytick.labelsize'] = 35
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['axes.linewidth'] = 2.5


# ==============================
# Benchmark Functions
# ==============================

def beale(x1, x2):
    """Beale function - multimodal test function with global minimum at (3, 0.5)"""
    return (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2


def quadratic(x1, x2):
    """Quadratic bowl function - convex test function with minimum at (0, 0)"""
    return 0.05 * x1 ** 2 + 5 * x2 ** 2


def rastrigin(x1, x2):
    """Rastrigin function - highly multimodal with many local minima"""
    A = 10
    if isinstance(x1, torch.Tensor):
        return A * 2 + (x1 ** 2 - A * torch.cos(2 * torch.pi * x1)) + (x2 ** 2 - A * torch.cos(2 * torch.pi * x2))
    else:
        return A * 2 + (x1 ** 2 - A * np.cos(2 * np.pi * x1)) + (x2 ** 2 - A * np.cos(2 * np.pi * x2))


def camel_back(x1, x2):
    """Six-hump camel back function - multiple local minima with global minima at (±0.0898, ∓0.7126)"""
    if isinstance(x1, torch.Tensor):
        return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
    else:
        return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2


def rosenbrock(x1, x2):
    """Rosenbrock function - banana-shaped valley with minimum at (1, 1)"""
    if isinstance(x1, torch.Tensor):
        return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2
    else:
        return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2


# ==============================
# Function Information Helpers
# ==============================

def get_optimal_point(func_name):
    """Return the global minimum point for each function"""
    optima = {
        "beale": np.array([3.0, 0.5]),
        "quadratic": np.array([0.0, 0.0]),
        "rastrigin": np.array([0.0, 0.0]),
        "camel_back": np.array([0.0898, -0.7126]),
        "rosenbrock": np.array([1.0, 1.0])
    }
    return optima.get(func_name)


def get_optimal_value(func_name):
    """Return the global minimum value for each function"""
    values = {
        "beale": beale(3.0, 0.5),
        "quadratic": quadratic(0.0, 0.0),
        "rastrigin": rastrigin(0.0, 0.0),
        "camel_back": camel_back(0.0898, -0.7126),
        "rosenbrock": rosenbrock(1.0, 1.0)
    }
    return values.get(func_name)


def get_visualization_range(func_name):
    """Return appropriate visualization ranges for each function"""
    ranges = {
        "beale": ([1, 5.0], [-2, 2]),
        "quadratic": ([-3, 1.0], [-2, 2]),
        "rastrigin": ([-5, 5], [-5, 5]),
        "camel_back": ([-2, 2], [-2, 2]),
        "rosenbrock": ([-2, 2], [-1, 3])
    }
    return ranges.get(func_name)


# ==============================
# Optimization Algorithms
# ==============================

def gd_optimizer(func, learning_rate, num_iterations, initial_point):
    """Gradient Descent optimizer"""
    x = torch.tensor(initial_point, requires_grad=True)
    iter_points = [x.detach().numpy().copy()]
    func_values = [func(x[0], x[1]).item()]

    for i in range(num_iterations):
        loss = func(x[0], x[1])
        loss.backward()

        with torch.no_grad():
            x -= learning_rate * x.grad
            iter_points.append(x.detach().numpy().copy())
            func_values.append(func(x[0], x[1]).item())
            x.grad.zero_()

    return np.array(iter_points), np.array(func_values)


def heavyball_optimizer(func, learning_rate, momentum, num_iterations, initial_point):
    """Heavyball (Momentum) optimizer"""
    x = torch.tensor(initial_point, requires_grad=True)
    v = torch.zeros(2)
    iter_points = [x.detach().numpy().copy()]
    func_values = [func(x[0], x[1]).item()]

    for i in range(num_iterations):
        loss = func(x[0], x[1])
        loss.backward()

        with torch.no_grad():
            v = momentum * v - learning_rate * x.grad
            x += v
            iter_points.append(x.detach().numpy().copy())
            func_values.append(func(x[0], x[1]).item())
            x.grad.zero_()

    return np.array(iter_points), np.array(func_values)


def nesterov_optimizer(func, learning_rate, momentum, num_iterations, initial_point):
    """Nesterov Accelerated Gradient optimizer"""
    x = torch.tensor(initial_point, requires_grad=True)
    v = torch.zeros(2)
    iter_points = [x.detach().numpy().copy()]
    func_values = [func(x[0], x[1]).item()]

    for i in range(num_iterations):
        x_future = x + momentum * v
        loss = func(x_future[0], x_future[1])
        loss.backward()

        with torch.no_grad():
            v = momentum * v - learning_rate * x.grad
            x += v
            iter_points.append(x.detach().numpy().copy())
            func_values.append(func(x[0], x[1]).item())
            x.grad.zero_()

    return np.array(iter_points), np.array(func_values)


def pid_optimizer(func, learning_rate, momentum, num_iterations, initial_point, gamma):
    """PID optimizer with derivative term"""
    x = torch.tensor(initial_point, requires_grad=True)
    v = torch.zeros(2)
    d = torch.zeros(2)
    prev_grad = None
    iter_points = [x.detach().numpy().copy()]
    func_values = [func(x[0], x[1]).item()]

    for i in range(num_iterations):
        loss = func(x[0], x[1])
        loss.backward()

        with torch.no_grad():
            if i == 0:
                v = momentum * v - learning_rate * x.grad
                x += v
                prev_grad = x.grad.clone()
            else:
                v = momentum * v - learning_rate * x.grad
                grad_diff = x.grad - prev_grad
                d = momentum * d - gamma * (1 - momentum) * grad_diff
                x += v + d
                prev_grad = x.grad.clone()

            iter_points.append(x.detach().numpy().copy())
            func_values.append(func(x[0], x[1]).item())
            x.grad.zero_()

    return np.array(iter_points), np.array(func_values)


def pd_optimizer(func, learning_rate, momentum, num_iterations, initial_point, gamma):
    """PD optimizer (Proportional-Derivative)"""
    x = torch.tensor(initial_point, requires_grad=True)
    v = torch.zeros(2)
    d = torch.zeros(2)
    prev_grad = None
    iter_points = [x.detach().numpy().copy()]
    func_values = [func(x[0], x[1]).item()]

    for i in range(num_iterations):
        loss = func(x[0], x[1])
        loss.backward()

        with torch.no_grad():
            if i == 0:
                x -= learning_rate * x.grad
                prev_grad = x.grad.clone()
            else:
                grad_diff = x.grad - prev_grad
                d = momentum * d - gamma * (1 - momentum) * grad_diff
                x = x - learning_rate * x.grad + d
                prev_grad = x.grad.clone()

            iter_points.append(x.detach().numpy().copy())
            func_values.append(func(x[0], x[1]).item())
            x.grad.zero_()

    return np.array(iter_points), np.array(func_values)


def adapid_optimizer(func, ada_learning_rate, momentum, num_iterations, initial_point, gamma):
    """Adaptive PID optimizer with automatic learning rate and momentum adjustment"""
    x = torch.tensor(initial_point, requires_grad=True)
    v = torch.zeros(2)
    d = torch.zeros(2)
    prev_x = None
    prev_grad = None

    iter_points = [x.detach().numpy().copy()]
    func_values = [func(x[0], x[1]).item()]
    adaptive_learning_rates = [ada_learning_rate]
    adaptive_momentums = [momentum]

    for i in range(num_iterations):
        loss = func(x[0], x[1])
        loss.backward()

        with torch.no_grad():
            if i == 0:
                v = momentum * v - ada_learning_rate * x.grad
                x += v
                prev_x = x.clone().detach()
                prev_grad = x.grad.clone()
                adaptive_learning_rates.append(ada_learning_rate)
                adaptive_momentums.append(momentum)
            else:
                # Adaptive learning rate calculation
                s_k = x - prev_x
                y_k = x.grad - prev_grad
                s_norm_sq = torch.sum(s_k ** 2)
                s_dot_y = torch.dot(s_k, y_k)

                if abs(s_dot_y.item()) > 1e-10:
                    numerator = ada_learning_rate * torch.min(s_norm_sq, s_dot_y)
                    adaptive_lr = numerator / s_dot_y
                    adaptive_lr = torch.clamp(adaptive_lr, ada_learning_rate / 10, ada_learning_rate * 10)
                else:
                    adaptive_lr = torch.tensor(ada_learning_rate)

                # Adaptive momentum calculation with constraint
                grad_norm_sq = torch.sum(x.grad ** 2)
                prev_grad_norm_sq = torch.sum(prev_grad ** 2)

                if prev_grad_norm_sq.item() > 1e-10:
                    fr_ratio = grad_norm_sq / prev_grad_norm_sq
                    adaptive_mom = torch.min(fr_ratio, torch.tensor(0.999))
                else:
                    adaptive_mom = torch.tensor(momentum)

                # Update parameters
                v = adaptive_mom * v - adaptive_lr * x.grad
                grad_diff = x.grad - prev_grad
                d = adaptive_mom * d - gamma * (1 - adaptive_mom) * grad_diff
                x += v + d
                prev_x = x.clone().detach()
                prev_grad = x.grad.clone()
                adaptive_learning_rates.append(adaptive_lr.item())
                adaptive_momentums.append(adaptive_mom.item())

            iter_points.append(x.detach().numpy().copy())
            func_values.append(func(x[0], x[1]).item())
            x.grad.zero_()

    return np.array(iter_points), np.array(func_values)


# ==============================
# Main Function
# ==============================

def main():
    # Configuration settings
    algorithms = ["HB", "NAG", "PID", "AdaPID"]  # Available algorithms
    function = "camel_back"  # Options: "beale", "quadratic", "rastrigin", "camel_back", "rosenbrock"

    # Parameter settings for different functions
    param_configs = {
        "beale": {
            "learning_rate": 0.0001, "ada_learning_rate": 0.0001, "gamma": 0.0005,
            "momentum": 0.9, "initial_point": [2.0, 2.0], "num_iterations": 500
        },
        "quadratic": {
            "learning_rate": 0.01, "ada_learning_rate": 0.01, "gamma": 0.05,
            "momentum": 0.9, "initial_point": [-2.0, 2.0], "num_iterations": 500
        },
        "rastrigin": {
            "learning_rate": 0.01, "ada_learning_rate": 0.01, "gamma": 0.01,
            "momentum": 0.9, "initial_point": [-4.0, 3.5], "num_iterations": 200
        },
        "camel_back": {
            "learning_rate": 0.01, "ada_learning_rate": 0.01, "gamma": 0.05,
            "momentum": 0.9, "initial_point": [-1.2, -1.75], "num_iterations": 200
        },
        "rosenbrock": {
            "learning_rate": 0.001, "ada_learning_rate": 0.001, "gamma": 0.001,
            "momentum": 0.9, "initial_point": [-1.5, 2.0], "num_iterations": 200
        }
    }

    # Get parameters for selected function
    params = param_configs[function]

    # Select the target function
    func_dict = {
        "beale": beale,
        "quadratic": quadratic,
        "rastrigin": rastrigin,
        "camel_back": camel_back,
        "rosenbrock": rosenbrock
    }
    func = func_dict[function]

    # Get optimal value for error calculation
    optimal_value = get_optimal_value(function)

    # Storage for results
    results = {}

    # Run optimization algorithms
    for algorithm in algorithms:
        if algorithm == "GD":
            iter_points, func_values = gd_optimizer(func, params["learning_rate"],
                                                    params["num_iterations"], params["initial_point"])
            algorithm_name = "GD"
            line_style = "solid"
            color = "blue"
            zorder = 4
        elif algorithm == "HB":
            iter_points, func_values = heavyball_optimizer(func, params["learning_rate"],
                                                           params["momentum"], params["num_iterations"],
                                                           params["initial_point"])
            algorithm_name = "HB"
            line_style = "dashed"
            color = "green"
            zorder = 2
        elif algorithm == "NAG":
            iter_points, func_values = nesterov_optimizer(func, params["learning_rate"],
                                                          params["momentum"], params["num_iterations"],
                                                          params["initial_point"])
            algorithm_name = "NAG"
            line_style = "dashdot"
            color = "gray"
            zorder = 1
        elif algorithm == "PID":
            iter_points, func_values = pid_optimizer(func, params["learning_rate"],
                                                     params["momentum"], params["num_iterations"],
                                                     params["initial_point"], params["gamma"])
            algorithm_name = "PID"
            line_style = "dotted"
            color = "purple"
            zorder = 3
        elif algorithm == "PD":
            iter_points, func_values = pd_optimizer(func, params["learning_rate"],
                                                    params["momentum"], params["num_iterations"],
                                                    params["initial_point"], params["gamma"])
            algorithm_name = "PD"
            line_style = "dotted"
            color = "orange"
            zorder = 5
        elif algorithm == "AdaPID":
            iter_points, func_values = adapid_optimizer(func, params["ada_learning_rate"],
                                                        params["momentum"], params["num_iterations"],
                                                        params["initial_point"], params["gamma"])
            algorithm_name = "AdaPID"
            line_style = "solid"
            color = "red"
            zorder = 0
        else:
            continue

        # Calculate optimization error
        errors = func_values - optimal_value

        # Store results
        results[algorithm] = {
            "iter_points": iter_points,
            "func_values": func_values,
            "errors": errors,
            "line_style": line_style,
            "color": color,
            "name": algorithm_name,
            "zorder": zorder
        }

    # Get visualization range
    x_range, y_range = get_visualization_range(function)

    # ==============================
    # Plot 1: Optimization Trajectories
    # ==============================
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Create contour plot
    x1_vals = np.linspace(x_range[0], x_range[1], 500)
    x2_vals = np.linspace(y_range[0], y_range[1], 500)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = func(X1, X2)

    # Enhanced contour settings for different functions
    if function == "quadratic":
        # Exponential spacing for quadratic function
        min_val, max_val = np.min(Z), np.max(Z)
        base, num_levels = 1.5, 20
        exp_levels = np.logspace(0, 1, num_levels, base=base) - 1
        exp_levels = exp_levels / exp_levels[-1] * (max_val - min_val) + min_val + 1e-10
        levels = np.clip(exp_levels, min_val + 1e-10, max_val)
        contour = ax1.contour(X1, X2, Z, levels=levels, colors='#1f77b4', alpha=0.5, linewidths=0.8, zorder=-1)
    else:
        # Linear spacing for other functions
        levels = np.linspace(np.min(Z), np.max(Z), 60)
        contour = ax1.contour(X1, X2, Z, levels=levels, colors='#1f77b4', alpha=0.5, linewidths=0.5, zorder=-1)

    # Plot optimization paths
    for algorithm in sorted(algorithms, key=lambda x: results[x]["zorder"]):
        data = results[algorithm]
        ax1.plot(data["iter_points"][:, 0], data["iter_points"][:, 1],
                 linestyle=data["line_style"], color=data["color"],
                 linewidth=3.5, label=data["name"], zorder=data["zorder"])

    # Mark optimal and initial points
    optimal_point = get_optimal_point(function)
    ax1.plot(optimal_point[0], optimal_point[1], 'o', color='red', markersize=5, label='Optimal Point', zorder=10)
    ax1.plot(params["initial_point"][0], params["initial_point"][1], '*', color='red', markersize=8,
             label='Initial Point', zorder=10)

    # Set plot properties
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_aspect('equal')
    ax1.legend()

    # Save contour plot
    plt.tight_layout()
    plt.savefig(f'Figures/Test_{function}.png', format='png', dpi=600,
                bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'Figures/Test_{function}.eps', format='eps',
                bbox_inches='tight', pad_inches=0.02, dpi=600)
    plt.close(fig1)

    # ==============================
    # Plot 2: Error Convergence
    # ==============================
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # Plot error convergence
    for algorithm in sorted(algorithms, key=lambda x: results[x]["zorder"]):
        data = results[algorithm]
        iterations = range(len(data["errors"]))
        ax2.plot(iterations, data["errors"],
                 linestyle=data["line_style"], color=data["color"],
                 linewidth=3.5, label=data["name"], zorder=data["zorder"])

    # Set axis limits based on function
    axis_settings = {
        "beale": {"xlim": (0, 100), "ylim": (1e-1, 1e3), "yticks": range(-1, 4)},
        "quadratic": {"xlim": (0, 125), "ylim": (1e-3, 1e2), "yticks": range(-3, 3)},
        "rastrigin": {"xlim": (0, 100), "ylim": (1e-3, 1e3), "yticks": range(-3, 4)},
        "camel_back": {"xlim": (0, 200), "ylim": (1e-7, 1e1), "yticks": range(-7, 1)},
        "rosenbrock": {"xlim": (0, 200), "ylim": (1e-3, 1e3), "yticks": range(-3, 4)}
    }

    settings = axis_settings[function]
    ax2.set_xlim(settings["xlim"])
    ax2.set_ylim(settings["ylim"])
    y_ticks = [10 ** i for i in settings["yticks"]]
    y_tick_labels = ['$10^{' + str(i) + '}$' for i in settings["yticks"]]
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_tick_labels)

    # Common settings
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('$f(x) - f(x^*)$')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Maintain aspect ratio
    x_min, x_max = ax2.get_xlim()
    y_min, y_max = ax2.get_ylim()
    y_min_log, y_max_log = np.log10(y_min), np.log10(y_max)
    data_ratio = (y_max_log - y_min_log) / (x_max - x_min)
    ax2.set_aspect(1.0 / data_ratio)

    # Save error plot
    plt.tight_layout()
    plt.savefig(f'Figures/Error_{function}.png', format='png', dpi=600,
                bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'Figures/Error_{function}.eps', format='eps',
                bbox_inches='tight', pad_inches=0.02, dpi=600)
    plt.close(fig2)

    # ==============================
    # Print Results Summary
    # ==============================
    print("Optimization Results Summary:")
    print("=" * 60)
    for algorithm in algorithms:
        if algorithm not in results:
            continue
        data = results[algorithm]
        final_point = data["iter_points"][-1]
        optimal_point = get_optimal_point(function)
        distance = np.linalg.norm(final_point - optimal_point)
        final_error = data["errors"][-1]

        print(f"{data['name']}:")
        print(f"  Final point: {final_point}")
        print(f"  Distance to optimum: {distance:.6f}")
        print(f"  Final error: {final_error:.6e}")
        print(f"  Total iterations: {len(data['errors']) - 1}")
        print()


if __name__ == "__main__":
    main()