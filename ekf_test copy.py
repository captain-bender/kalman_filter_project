from simulation_ekf import generate_diff_drive_trajectory
from ekf_filter import EKFDD
import matplotlib.pyplot as plt
import numpy as np

def run_ekf_synthetic():
    timesteps = 100
    dt = 1.0
    process_var = [0.01, 0.01, 0.001]
    measurement_var = [0.25, 0.5]
    init_P = [100, 100, 5]

    true_states, controls, measurements = generate_diff_drive_trajectory(
        timesteps=timesteps,
        dt=dt,
        process_noise_std=(0.01, 0.01, 0.01),
        measurement_noise_std=np.sqrt(measurement_var)
    )

    ekf = EKFDD(
        dt=dt,
        process_var=process_var,
        measurement_var=measurement_var,
        init_P=init_P,
    )
    # Use this if your robot starts at 0,0,0
    ekf.initialize([0, 0, 0])
    # Or initialize using the first measurement (sometimes better)
    # ekf.initialize([measurements[0][0], measurements[0][1], 0])

    est_states = []
    for k in range(timesteps):
        v, w = controls[k]         # [linear vel, angular vel]
        z = measurements[k]        # [x_measured, y_measured]
        ekf.predict([v, w])
        ekf.update(z)
        est_states.append(ekf.get_state())
    est_states = np.array(est_states)
    return true_states, controls, measurements, est_states

def plot_ekf_results(true_states, measurements, est_states):
    plt.figure(figsize=(8, 6))
    plt.plot(true_states[:,0], true_states[:,1], label='True Trajectory', linewidth=2)
    plt.scatter(measurements[:,0], measurements[:,1], color='red', s=25, label='Measurements', alpha=0.5)
    plt.plot(est_states[:,0], est_states[:,1], color='green', label='EKF Estimate', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.title('EKF Synthetic Robot Tracking')
    plt.show()

if __name__ == "__main__":
    true_states, controls, measurements, est_states = run_ekf_synthetic()
    print("First 5 EKF position estimates:\n", est_states[:5])
    plot_ekf_results(true_states, measurements, est_states)
