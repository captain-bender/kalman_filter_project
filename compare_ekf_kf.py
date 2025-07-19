import numpy as np
import matplotlib.pyplot as plt
from simulation_ekf import generate_diff_drive_trajectory
from ekf_filter import EKFDD
from kalman_filter import KalmanFilterCV

def run_ekf_and_kf_comparison():
    # --- Parameters ---
    timesteps = 100
    dt = 1.0
    process_var_ekf = [0.01, 0.01, 0.001]   # EKF Q for [x, y, theta]
    measurement_var_ekf = [0.25, 0.5]       # EKF R for [x, y]
    init_P_ekf = [100, 100, 5]              # EKF P for [x, y, theta]

    process_var_kf = 0.01                   # KF Q (use scalar for simplicity)
    measurement_var_kf = 0.5                # KF R (use scalar for simplicity)
    init_vel_kf = [0, 0]                    # If you want, set to [true_vx, true_vy]

    # --- Data Generation ---
    true_states, controls, measurements = generate_diff_drive_trajectory(
        timesteps=timesteps,
        dt=dt,
        process_noise_std=(0.01, 0.01, 0.01),
        measurement_noise_std=np.sqrt(measurement_var_ekf)
    )

    # --- EKF ---
    ekf = EKFDD(
        dt=dt,
        process_var=process_var_ekf,
        measurement_var=measurement_var_ekf,
        init_P=init_P_ekf,
    )
    # You may also use measurements[0][0:2] as start point.
    ekf.initialize([0, 0, 0])

    est_states_ekf = []
    for k in range(timesteps):
        v, w = controls[k]
        z = measurements[k]
        ekf.predict([v, w])
        ekf.update(z)
        est_states_ekf.append(ekf.get_state())
    est_states_ekf = np.array(est_states_ekf)

    # --- Linear Kalman Filter (Constant Velocity Model) ---
    kf = KalmanFilterCV(
        dt=dt,
        process_var=process_var_kf,
        measurement_var=measurement_var_kf,
    )
    # If you want, use the first displacement to estimate initial velocity.
    kf.initialize(
        init_pos=measurements[0],
        init_vel=init_vel_kf
    )

    est_states_kf = []
    for z in measurements:
        kf.predict()
        kf.update(z)
        est_states_kf.append(kf.get_state())
    est_states_kf = np.array(est_states_kf)   # shape: (timesteps, 4), cols: [x, y, vx, vy]

    return true_states, measurements, est_states_ekf, est_states_kf

def rmse(true_xy, est_xy):
    """
    Compute Root Mean Squared Error (only for (x, y) columns).
    Args:
        true_xy:     np.ndarray shape (N, 2) true positions
        est_xy:      np.ndarray shape (N, 2) estimated positions
    Returns:
        float: RMSE value
    """
    error = true_xy - est_xy
    mse = np.mean(np.sum(error**2, axis=1))
    return np.sqrt(mse)

def mae(true_xy, est_xy):
    """
    Compute Mean Absolute Error (only for (x, y) columns).
    Args:
        true_xy:     np.ndarray shape (N, 2) true positions
        est_xy:      np.ndarray shape (N, 2) estimated positions
    Returns:
        float: MAE value
    """
    abs_error = np.abs(true_xy - est_xy)
    return np.mean(np.sum(abs_error, axis=1))

def per_step_errors(true_states, est_states):
    """
    Returns per-timestep RMSE and MAE arrays.
    """
    error = true_states[:, :2] - est_states[:, :2]
    per_step_rmse = np.sqrt(np.sum(error**2, axis=1))
    per_step_mae = np.sum(np.abs(error), axis=1)
    return per_step_rmse, per_step_mae

def plot_heading_comparison(true_states, est_states_ekf):
    time = np.arange(true_states.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(time, true_states[:, 2], label='True Heading (θ)', color='blue', linewidth=2)
    plt.plot(time, est_states_ekf[:, 2], label='EKF Estimated Heading (θ)', color='green', linestyle='--', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Heading (radians)')
    plt.title('True vs. EKF-Estimated Robot Heading')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_comparison(true_states, measurements, est_states_ekf, est_states_kf):
    plt.figure(figsize=(9,7))
    plt.plot(true_states[:,0], true_states[:,1], label='True Trajectory', linewidth=2, color='blue')
    plt.scatter(measurements[:,0], measurements[:,1], color='red', s=25, label='Measurements', alpha=0.3)
    plt.plot(est_states_ekf[:,0], est_states_ekf[:,1], color='green', label='EKF Estimate', linewidth=2)
    plt.plot(est_states_kf[:,0], est_states_kf[:,1], color='orange', linestyle='--', linewidth=2, label='Linear KF Estimate')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Differential-Drive Robot: EKF vs Linear KF')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_filter_comparison(true_states, measurements, est_states_ekf, est_states_kf, ekf_rmse, kf_rmse):
    plt.figure(figsize=(10, 7))
    plt.plot(true_states[:,0], true_states[:,1], label='True Trajectory', linewidth=2, color='blue')
    plt.scatter(measurements[:,0], measurements[:,1], color='red', s=25, label='Measurements', alpha=0.3)
    plt.plot(est_states_ekf[:,0], est_states_ekf[:,1], color='green', label=f'EKF Estimate (RMSE={ekf_rmse:.3f})', linewidth=2)
    plt.plot(est_states_kf[:,0], est_states_kf[:,1], color='orange', linestyle='--', linewidth=2, label=f'Linear KF Estimate (RMSE={kf_rmse:.3f})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Differential-Drive Robot: EKF vs Linear KF')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_error_over_time(per_step_ekf_rmse, per_step_kf_rmse, per_step_ekf_mae, per_step_kf_mae):
    timesteps = len(per_step_ekf_rmse)
    time = np.arange(timesteps)

    plt.figure(figsize=(14, 6))

    # RMSE plot
    plt.subplot(1, 2, 1)
    plt.plot(time, per_step_ekf_rmse, label='EKF RMSE', color='green')
    plt.plot(time, per_step_kf_rmse, label='Linear KF RMSE', color='orange', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Euclidean Error (RMSE)')
    plt.title('Position RMSE Over Time')
    plt.legend()
    plt.grid(True)

    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(time, per_step_ekf_mae, label='EKF MAE', color='green')
    plt.plot(time, per_step_kf_mae, label='Linear KF MAE', color='orange', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Absolute Error (MAE)')
    plt.title('Position MAE Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    true_states, measurements, est_states_ekf, est_states_kf = run_ekf_and_kf_comparison()
    print("First 5 EKF position estimates:\n", est_states_ekf[:5, :2])
    print("First 5 Linear KF position estimates:\n", est_states_kf[:5, :2])
    # plot_comparison(true_states, measurements, est_states_ekf, est_states_kf)

    ekf_rmse = rmse(true_states[:, :2], est_states_ekf[:, :2])
    kf_rmse  = rmse(true_states[:, :2], est_states_kf[:, :2])
    ekf_mae  = mae(true_states[:, :2], est_states_ekf[:, :2])
    kf_mae   = mae(true_states[:, :2], est_states_kf[:, :2])

    print(f"EKF RMSE: {ekf_rmse:.4f}")
    print(f"Linear KF RMSE: {kf_rmse:.4f}")
    print(f"EKF MAE: {ekf_mae:.4f}")
    print(f"Linear KF MAE: {kf_mae:.4f}")

    # Display as a table
    print("\n| Filter    | RMSE    | MAE     |")
    print("|-----------|---------|---------|")
    print(f"| EKF       | {ekf_rmse:.4f} | {ekf_mae:.4f} |")
    print(f"| Linear KF | {kf_rmse:.4f} | {kf_mae:.4f} |")

    # plot_filter_comparison(true_states, measurements, est_states_ekf, est_states_kf, ekf_rmse, kf_rmse)

    # Per-step errors
    per_step_ekf_rmse, per_step_ekf_mae = per_step_errors(true_states, est_states_ekf)
    per_step_kf_rmse, per_step_kf_mae   = per_step_errors(true_states, est_states_kf)

    # plot_error_over_time(per_step_ekf_rmse, per_step_kf_rmse, per_step_ekf_mae, per_step_kf_mae)
    plot_heading_comparison(true_states, est_states_ekf)
    
