import numpy as np
import matplotlib.pyplot as plt
from simulation import generate_2d_trajectory

def plot_trajectory(true_positions, measurements):
    plt.figure(figsize=(8, 6))
    plt.plot(true_positions[:, 0], true_positions[:, 1], label='True trajectory', linewidth=2)
    plt.scatter(measurements[:, 0], measurements[:, 1], label='Noisy measurements', color='red', s=25, alpha=0.6)
    plt.title("2D Trajectory: Ground Truth vs. Measurements")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_kalman_results(true_positions, measurements, estimated_states):
    """
    Plots the true trajectory, noisy measurements, and Kalman filter estimates.

    Args:
        true_positions (np.ndarray): Ground truth positions, shape (N, 2)
        measurements (np.ndarray): Noisy measurements, shape (N, 2)
        estimated_states (np.ndarray): Kalman filter states, shape (N, 4)
    """
    # Extract x and y components for plotting
    x_true, y_true = true_positions[:, 0], true_positions[:, 1]
    x_meas, y_meas = measurements[:, 0], measurements[:, 1]
    x_est, y_est = estimated_states[:, 0], estimated_states[:, 1]

    plt.figure(figsize=(8, 6))
    plt.plot(x_true, y_true, label="True trajectory", color='blue', linewidth=2)
    plt.scatter(x_meas, y_meas, label="Noisy measurements", color='red', s=30, alpha=0.5)
    plt.plot(x_est, y_est, label="Kalman filter estimate", color='green', linewidth=2)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("2D Trajectory: Ground Truth, Measurements, and Kalman Estimate")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    true_positions, measurements = generate_2d_trajectory()
    plot_trajectory(true_positions, measurements)
