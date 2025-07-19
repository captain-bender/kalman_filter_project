import numpy as np

def generate_2d_trajectory(
        timesteps=100, 
        dt=1.0, 
        init_pos=(0, 0), 
        velocity=(1, 0.5), 
        process_noise_std=0.0,   # No process noise for simplicity
        measurement_noise_std=1.0
    ):
    """
    Generate a 2D trajectory with constant velocity and noisy measurements.

    Returns:
        true_positions: np.ndarray of shape (timesteps, 2)
        measurements: np.ndarray of shape (timesteps, 2)
    """
    true_positions = np.zeros((timesteps, 2))
    measurements = np.zeros((timesteps, 2))
    pos = np.array(init_pos, dtype=float)
    vel = np.array(velocity, dtype=float)

    for t in range(timesteps):
        # Optionally add process noise to the true trajectory
        process_noise = np.random.randn(2) * process_noise_std
        pos = pos + vel * dt + process_noise

        true_positions[t] = pos

        # Add measurement noise
        measurement_noise = np.random.randn(2) * measurement_noise_std
        measurements[t] = pos + measurement_noise

    return true_positions, measurements

if __name__ == "__main__":
    true_positions, measurements = generate_2d_trajectory()
    print("True positions (first 5):\n", true_positions[:5])
    print("Measured positions (first 5):\n", measurements[:5])
