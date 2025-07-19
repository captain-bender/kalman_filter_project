# Kalman Filter Project

A comprehensive implementation and comparison of Kalman Filter (KF) and Extended Kalman Filter (EKF) algorithms for tracking and state estimation.

## Overview

This project implements two types of Kalman filters:
- **Linear Kalman Filter**: For constant velocity motion model in 2D
- **Extended Kalman Filter**: For differential drive robot motion with nonlinear dynamics

The project includes simulation environments, visualization tools, and comparison utilities to demonstrate the effectiveness of both filtering approaches.

## Features

- **Linear Kalman Filter** (`kalman_filter.py`)
  - 2D constant velocity motion model
  - State vector: [x, y, vx, vy]
  - Handles position measurements with velocity estimation

- **Extended Kalman Filter** (`ekf_filter.py`)  
  - Differential drive robot model
  - State vector: [x, y, theta]
  - Nonlinear motion model with linearized observations

- **Simulation Environments**
  - 2D trajectory generation with configurable noise
  - Differential drive robot trajectory simulation
  - Customizable process and measurement noise parameters

- **Visualization Tools**
  - Real-time plotting of trajectories, measurements, and estimates
  - Comparison plots between different filter approaches
  - Error analysis and performance metrics

## File Structure

```
kalman_filter_project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
├── kalman_filter.py            # Linear Kalman Filter implementation
├── ekf_filter.py               # Extended Kalman Filter implementation
├── simulation.py               # 2D trajectory simulation
├── simulation_ekf.py           # Differential drive simulation
├── plot_results.py             # Visualization utilities
├── compare_ekf_kf.py           # EKF vs KF comparison
├── kalman_test.py              # Kalman Filter tests
├── ekf_test.py                 # EKF tests
├── ekf_test copy.py            # Additional EKF tests
├── data/                       # Data directory (for future datasets)
└── tests/                      # Test directory (for future unit tests)
```

## Requirements

- Python 3.7+
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- pytest >= 6.0.0 (for running tests)

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy matplotlib pytest
```

## Usage

### Running Basic Kalman Filter Example

```bash
python kalman_test.py
```

### Running Extended Kalman Filter Example

```bash
python ekf_test.py
```

### Comparing EKF vs Linear KF

```bash
python compare_ekf_kf.py
```

### Custom Simulation

```python
from kalman_filter import KalmanFilterCV
from simulation import generate_2d_trajectory
import numpy as np

# Generate trajectory
true_pos, measurements = generate_2d_trajectory(
    timesteps=100,
    dt=1.0,
    velocity=(1, 0.5),
    measurement_noise_std=1.0
)

# Initialize Kalman Filter
kf = KalmanFilterCV(dt=1.0, process_var=0.01, measurement_var=1.0)
kf.initialize(measurements[0])

# Run filter
estimates = []
for measurement in measurements:
    kf.predict()
    kf.update(measurement)
    estimates.append(kf.x[:2].flatten())
```

## Algorithm Details

### Linear Kalman Filter

The linear Kalman filter assumes a constant velocity motion model:

**State Transition:**
```
x(k+1) = F * x(k) + w(k)
```

**Measurement Model:**
```
z(k) = H * x(k) + v(k)
```

Where:
- State: [x, y, vx, vy]
- Measurements: [x, y] (position only)
- Process noise w(k) ~ N(0, Q)
- Measurement noise v(k) ~ N(0, R)

### Extended Kalman Filter

The EKF handles nonlinear motion models using linearization:

**State:** [x, y, theta] (position and orientation)

**Nonlinear Motion Model:**
```
x(k+1) = f(x(k), u(k)) + w(k)
z(k) = h(x(k)) + v(k)
```

The filter linearizes the motion and measurement models using Jacobians for the predict and update steps.

## Parameters

### Kalman Filter Parameters
- `dt`: Time step (seconds)
- `process_var`: Process noise variance
- `measurement_var`: Measurement noise variance

### EKF Parameters
- `dt`: Time step (seconds)
- `process_var`: Process noise variance (can be scalar or [x, y, theta])
- `measurement_var`: Measurement noise variance (can be scalar or [x, y])
- `init_P`: Initial state uncertainty

## Examples and Results

The project includes several test scripts that demonstrate:

1. **Basic tracking performance** with different noise levels
2. **Comparison between linear and nonlinear models**
3. **Parameter sensitivity analysis**
4. **Visualization of estimation accuracy**

Run the test scripts to see plots comparing:
- True trajectory vs. noisy measurements
- Filter estimates vs. ground truth
- Estimation errors over time
- Filter performance metrics

## Contributing

Feel free to contribute by:
- Adding new motion models
- Implementing additional filter variants (UKF, Particle Filter)
- Adding more comprehensive test cases
- Improving visualization tools
- Adding real-world datasets

## License

This project is open source. Feel free to use and modify for educational and research purposes.

## References

- Kalman, R.E. "A New Approach to Linear Filtering and Prediction Problems" (1960)
- Julier, S.J. and Uhlmann, J.K. "Unscented Filtering and Nonlinear Estimation" (2004)
- Thrun, S., Burgard, W., and Fox, D. "Probabilistic Robotics" (2005)
