import numpy as np

def generate_diff_drive_trajectory(
    timesteps=100,
    dt=1.0,
    init_state=(0, 0, 0),     # (x, y, theta)
    input_func=None,
    process_noise_std=(0.01, 0.01, 0.01),    # Noise std for (x, y, theta)
    measurement_noise_std=1.0
):
    """
    Simulate a differential drive robot with (v, w) commands.

    Returns:
        true_states: (N, 3) ndarray
        controls: (N, 2) ndarray
        measurements: (N, 2) ndarray - only (x, y) measured
    """
    x = np.array(init_state, dtype=float)
    true_states = np.zeros((timesteps, 3))  # (x, y, theta)
    measurements = np.zeros((timesteps, 2)) # (x, y)
    controls = np.zeros((timesteps, 2))     # (v, w)

    for t in range(timesteps):
        if input_func is not None:
            v, w = input_func(t*dt, x.copy())
        else:
            v = 1.0         # constant forward
            w = 0.1 if t > timesteps // 2 else 0.0   # start turning halfway
        controls[t] = [v, w]

        # True state update (motion model)
        noise = np.random.randn(3)
        dx = v * np.cos(x[2]) * dt + process_noise_std[0] * noise[0]
        dy = v * np.sin(x[2]) * dt + process_noise_std[1] * noise[1]
        dtheta = w * dt + process_noise_std[2] * noise[2]
        x[0] += dx
        x[1] += dy
        x[2] += dtheta
        x[2] = (x[2] + np.pi) % (2 * np.pi) - np.pi  # keep theta in [-pi, pi]

        true_states[t] = x.copy()

        # Simulated noisy measurement (of x, y only)
        meas_noise = np.random.randn(2) * measurement_noise_std
        measurements[t] = x[:2] + meas_noise

    return true_states, controls, measurements

if __name__ == "__main__":
    # Example usage
    true_states, controls, measurements = generate_diff_drive_trajectory()
    print("First 5 ground truth states:\n", true_states[:5])
    print("First 5 control inputs (v, w):\n", controls[:5])
    print("First 5 position measurements:\n", measurements[:5])
