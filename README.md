# Optimization Algorithms Benchmark Suite
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
Official implementation of optimization algorithms from the paper: 
**"AdaPID: Adaptive Momentum Gradient Method based on PID Controller for Non-Convex Stochastic Optimization in Deep Learning"**
*Ailun Jian, Xun Li, Weigang Sun, Gaohang Yu* 

# Project Structure
├── Test_Function/                 # Benchmark functions optimization
│   └── main.py                   # Test functions with various optimizers
├── Image_classification/         # Real-world applications
│   ├── optimizers/              # Custom optimizer implementations
│   │   ├── AdaSPID.py          # Adaptive Stochastic PID optimizer
│   │   └── FRSGDM.py           # Fletcher-Reeves SGD with Momentum
│   ├── MNIST.py                # MNIST dataset training
│   └── Fashion_MNIST.py        # Fashion-MNIST dataset training
└── README.md

# Citation
@article{jian2025adapid,
  title={AdaPID: Adaptive Momentum Gradient Method based on PID Controller for Non-Convex Stochastic Optimization in Deep Learning},
  author={Jian, Ailun and Li, Xun and Sun, Weigang and Yu, Gaohang},
  journal={Authorea Preprints},
  year={2025},
  publisher={Authorea}
}
