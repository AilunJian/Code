import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import *
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import time
import argparse
import numpy as np
from AdaSPID import AdaSPID
from FRSGDM import FRSGDM
from utils import Logger, AverageMeter, accuracy, mkdir_p

# Parameter configuration
parser = argparse.ArgumentParser(description='MNIST Example with Adam, SGDM, AdaSPID and FRSGDM')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Adam')
parser.add_argument('--sgdm-lr', type=float, default=0.01, help='learning rate for SGDM and FRSGDM')
parser.add_argument('--ada-lr', type=float, default=0.1, help='learning rate for AdaSPID')
parser.add_argument('--rmsprop-lr', type=float, default=0.001, help='learning rate for RMSprop')
parser.add_argument('--adamw-lr', type=float, default=0.001, help='learning rate for AdamW')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGDM and AdaSPID')
parser.add_argument('--gamma', type=float, default=0.01, help='gamma for AdaSPID')
parser.add_argument('--fr-min-momentum', type=float, default=0.1, help='min momentum for FRSGDM')
parser.add_argument('--fr-max-momentum', type=float, default=0.999, help='max momentum for FRSGDM')
parser.add_argument('--gpu', type=int, default=0, help='the number of gpu for training')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--model', type=str, default='FNN', choices=['FNN', 'CNN'], help='model type')
parser.add_argument('--n-runs', type=int, default=5, help='number of independent runs')
parser.add_argument('--confidence-level', type=float, default=0.9, help='confidence level for intervals')
parser.add_argument('--use-scheduler', action='store_true', help='learning rate scheduler')
args = parser.parse_args()

# Dataset and data loaders
training_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)


# Neural network model definitions
class Net(nn.Module):
    """Fully connected neural network for MNIST classification"""

    def __init__(self, input_size=28 * 28, hidden_size=1000, output_size=10):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


class CNet(nn.Module):
    """Convolutional neural network for MNIST classification"""

    def __init__(self):
        super(CNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5, stride=1)
        self.maxpooling1 = nn.MaxPool2d(2)
        self.maxpooling2 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(360, 150)
        self.fc2 = nn.Linear(150, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        in_size = x.size(0)
        x = self.relu(self.maxpooling1(self.conv1(x)))
        x = self.relu(self.maxpooling2(self.conv2(x)))
        x = x.view(in_size, -1)
        x = self.dropout(self.fc1(x))
        return self.fc2(x)


# Training loop
def train_loop(train_data, model, loss_fn, optimizer, epoch, device, scheduler=None):
    """Training loop for one epoch"""
    model.train()
    train_loss_log = AverageMeter()
    train_acc_log = AverageMeter()

    for batch, (images, labels) in enumerate(train_data):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        train_loss = loss_fn(outputs, labels)
        train_loss.backward()
        optimizer.step()

        prec1, _ = accuracy(outputs.data, labels.data, topk=(1, 5))
        train_loss_log.update(train_loss.item(), images.size(0))
        train_acc_log.update(prec1.item(), images.size(0))

        if (batch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{args.num_epochs}], Step [{batch + 1}/{len(train_data)}], '
                  f'Loss: {train_loss_log.avg:.4f}, Acc: {train_acc_log.avg:.4f}')

    if scheduler and args.use_scheduler:
        scheduler.step()

    return train_loss_log, train_acc_log


# Testing loop
def test_loop(test_data, model, loss_fn, device):
    """Testing loop for model evaluation"""
    val_loss_log = AverageMeter()
    val_acc_log = AverageMeter()
    model.eval()

    with torch.no_grad():
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss = loss_fn(outputs, labels)
            val_loss_log.update(test_loss.item(), images.size(0))
            prec1, _ = accuracy(outputs.data, labels.data, topk=(1, 5))
            val_acc_log.update(prec1.item(), images.size(0))

    print(f'Test Accuracy: {val_acc_log.avg:.4f}%, Test Loss: {val_loss_log.avg:.4f}')
    return val_loss_log, val_acc_log


# Optimizer selection function
def get_optimizer(model, optimizer_name, learning_rate, momentum=0.9, gamma=0.01):
    """Get optimizer instance based on name"""
    if optimizer_name == 'Adam':
        return Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_name == 'SGDM':
        return SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4)
    elif optimizer_name == 'AdaSPID':
        return AdaSPID(model.parameters(), lr=learning_rate, momentum=momentum, gamma=gamma, weight_decay=1e-4)
    elif optimizer_name == 'FRSGDM':
        return FRSGDM(model.parameters(), lr=learning_rate, momentum=momentum,
                      min_momentum=args.fr_min_momentum, max_momentum=args.fr_max_momentum,
                      weight_decay=1e-4)
    # Add RMSprop and AdamW
    elif optimizer_name == 'RMSprop':
        return RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_name == 'AdamW':
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# Multi-run logger class
class MultiRunLogger:
    """Logger for multiple independent runs with statistical analysis"""

    def __init__(self, base_path, optimizer_name, model_name, n_runs, confidence_level=0.9):
        self.base_path = base_path
        self.optimizer_name = optimizer_name
        self.model_name = model_name
        self.n_runs = n_runs
        self.confidence_level = confidence_level
        self.all_run_data = []
        # Keep only two losses and two accuracies
        self.metrics = ['Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Time']

        self.optimizer_path = os.path.join(base_path, optimizer_name)
        if not os.path.exists(self.optimizer_path):
            os.makedirs(self.optimizer_path)

    def add_run(self, run_idx, data):
        """Add data from a single run"""
        # Save single run results
        run_logger = Logger(os.path.join(self.optimizer_path, f'run_{run_idx}_{self.model_name}.txt'))
        run_logger.set_names(self.metrics)
        for row in data:
            run_logger.append(row)
        run_logger.close()

        # Save to memory for statistical calculation
        self.all_run_data.append(data)

    def calculate_statistics(self):
        """Calculate statistics across all runs"""
        if not self.all_run_data:
            return None, None

        n_epochs = len(self.all_run_data[0])
        n_metrics = len(self.metrics)

        # Initialize result arrays
        means = np.zeros((n_epochs, n_metrics))
        confidence_intervals = np.zeros((n_epochs, n_metrics, 2))  # lower and upper bounds

        # Convert to numpy array for calculation
        all_data = np.array(self.all_run_data)  # shape: (n_runs, n_epochs, n_metrics)

        # Calculate statistics for each epoch and each metric
        for epoch in range(n_epochs):
            for metric in range(n_metrics):
                values = all_data[:, epoch, metric]

                # Check and handle NaN values
                valid_values = values[~np.isnan(values)]
                if len(valid_values) == 0:
                    # All values are NaN
                    means[epoch, metric] = np.nan
                    confidence_intervals[epoch, metric, 0] = np.nan
                    confidence_intervals[epoch, metric, 1] = np.nan
                    continue

                # Calculate mean
                means[epoch, metric] = np.mean(valid_values)

                # Calculate confidence interval
                if len(valid_values) > 1:
                    # Calculate standard deviation and standard error
                    std_dev = np.std(valid_values, ddof=1)  # Use ddof=1 for sample standard deviation
                    std_err = std_dev / np.sqrt(len(valid_values))

                    # Check if standard error is valid
                    if np.isfinite(std_err) and std_err > 0:
                        try:
                            from scipy import stats
                            ci = stats.t.interval(
                                self.confidence_level,
                                len(valid_values) - 1,
                                loc=means[epoch, metric],
                                scale=std_err
                            )
                            confidence_intervals[epoch, metric, 0] = ci[0]
                            confidence_intervals[epoch, metric, 1] = ci[1]
                        except:
                            # If calculation fails, use mean as confidence interval
                            confidence_intervals[epoch, metric, 0] = means[epoch, metric]
                            confidence_intervals[epoch, metric, 1] = means[epoch, metric]
                    else:
                        # Invalid standard error, use mean as confidence interval
                        confidence_intervals[epoch, metric, 0] = means[epoch, metric]
                        confidence_intervals[epoch, metric, 1] = means[epoch, metric]
                else:
                    # Only one valid value, cannot calculate confidence interval
                    confidence_intervals[epoch, metric, 0] = means[epoch, metric]
                    confidence_intervals[epoch, metric, 1] = means[epoch, metric]

        return means, confidence_intervals

    def save_statistics(self, means, confidence_intervals):
        """Save statistical results"""
        stats_logger = Logger(os.path.join(self.optimizer_path, f'stats_{self.model_name}.txt'))
        # Set column names
        column_names = [f'{metric} Mean' for metric in self.metrics] + \
                       [f'{metric} CI Lower' for metric in self.metrics] + \
                       [f'{metric} CI Upper' for metric in self.metrics]

        stats_logger.set_names(column_names)

        for epoch in range(len(means)):
            row = list(means[epoch]) + \
                  list(confidence_intervals[epoch, :, 0]) + \
                  list(confidence_intervals[epoch, :, 1])
            stats_logger.append(row)

        stats_logger.close()


# Main training function
def main(train_data, test_data, model, loss_fn, optimizer, num_epochs, device, scheduler=None):
    """Modified main training function that returns data for all epochs"""
    epoch_data = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        start_time = time.time()

        train_loss_log, train_acc_log = train_loop(
            train_data=train_data,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            scheduler=scheduler
        )

        end_time = time.time()
        execution_time = round(end_time - start_time, 3)

        val_loss_log, val_acc_log = test_loop(
            test_data=test_data,
            model=model,
            loss_fn=loss_fn,
            device=device
        )

        # Collect only two losses and two accuracy data
        epoch_data.append([
            train_loss_log.avg,
            val_loss_log.avg,
            train_acc_log.avg,
            val_acc_log.avg,
            execution_time  # Add time data
        ])

    return epoch_data


if __name__ == '__main__':
    # Create results directory
    path = 'results/mnist'
    if not os.path.exists(path):
        os.makedirs(path)

    # Select model
    NN_set = {'FNN': Net(), 'CNN': CNet()}
    initial_net = NN_set[args.model]

    # Save model structure
    model_path = os.path.join(path, f'{args.model}_model.pkl')
    torch.save(initial_net, model_path)

    # Optimizer list
    optimizers = ['Adam', 'SGDM', 'AdaSPID', 'FRSGDM', 'RMSprop', 'AdamW']
    learning_rates = {
        'Adam': args.lr,
        'SGDM': args.sgdm_lr,
        'AdaSPID': args.ada_lr,
        'FRSGDM': args.sgdm_lr,  # Use same learning rate as SGDM
        'RMSprop': args.rmsprop_lr,
        'AdamW': args.adamw_lr
    }

    # Device selection
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train each optimizer multiple times
    for optimizer_name in optimizers:
        print(f'\n{"=" * 50}')
        print(f'Training with {optimizer_name} optimizer ({args.n_runs} runs)')
        print(f'{"=" * 50}')

        # Initialize multi-run logger
        multi_logger = MultiRunLogger(path, optimizer_name, args.model, args.n_runs, args.confidence_level)

        for run_idx in range(args.n_runs):
            print(f'\n--- Run {run_idx + 1}/{args.n_runs} ---')

            # Set random seed for independence
            seed = 42 + run_idx * 100
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Load model
            model = torch.load(model_path)
            model.to(device)

            loss_fn = nn.CrossEntropyLoss()

            # Optimizer
            optimizer = get_optimizer(
                model,
                optimizer_name,
                learning_rates[optimizer_name],
                momentum=args.momentum,
                gamma=args.gamma
            )

            # Learning rate scheduler
            scheduler = None
            if args.use_scheduler:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

            # Training and testing
            run_data = main(
                train_data=train_loader,
                test_data=test_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                num_epochs=args.num_epochs,
                device=device,
                scheduler=scheduler
            )

            # Save this run's data
            multi_logger.add_run(run_idx, run_data)

        # Calculate statistics and save
        means, confidence_intervals = multi_logger.calculate_statistics()
        if means is not None:
            multi_logger.save_statistics(means, confidence_intervals)

        print(f'\nCompleted {args.n_runs} runs for {optimizer_name}')