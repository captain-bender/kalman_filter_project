import numpy as np
from simulation import generate_2d_trajectory
from kalman_filter import KalmanFilterCV
from plot_results import plot_kalman_results

def run_kalman_on_synthetic():
    timesteps = 100
    dt = 1.0
    process_var = 0.01
    measurement_var = 1.0

    true_positions, measurements = generate_2d_trajectory(
        timesteps=timesteps, dt=dt, process_noise_std=0.0, measurement_noise_std=np.sqrt(measurement_var)
    )

    kf = KalmanFilterCV(dt, process_var, measurement_var)
    kf.initialize(init_pos=measurements[0])  # Start filter at first measurement

    estimated_states = []
    for z in measurements:
        kf.predict()
        kf.update(z)
        estimated_states.append(kf.get_state())
    estimated_states = np.array(estimated_states)

    return true_positions, measurements, estimated_states

if __name__ == "__main__":
    true_positions, measurements, estimated_states = run_kalman_on_synthetic()
    # From here, pass this to your plotting routine
    print("First 5 Kalman estimates:\n", estimated_states[:5])
    plot_kalman_results(true_positions, measurements, estimated_states)
