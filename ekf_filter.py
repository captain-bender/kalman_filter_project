import numpy as np

class EKFDD:
    def __init__(
        self,
        dt,
        process_var=0.01,
        measurement_var=1.0,
        init_P=500.0
    ):
        """
        Extended Kalman Filter for a differential drive robot.
        State: [x, y, theta]

        Args:
            dt (float): Time step duration
            process_var (float|list|array, optional): Process noise variance (can be scalar or 3-element list for [x, y, theta])
            measurement_var (float|list|array, optional): Measurement noise variance (can be scalar or 2-element list for [x, y])
            init_P (float|list|array, optional): Initial state uncertainty (scalar or 3-element vector)
        """

        self.dt = dt
        self.x = np.zeros((3, 1))  # [x; y; theta]

        # Initial Covariance
        if isinstance(init_P, (list, tuple, np.ndarray)):
            self.P = np.diag(np.array(init_P))
        else:
            self.P = np.eye(3) * init_P

        # Process Noise Covariance
        if isinstance(process_var, (list, tuple, np.ndarray)):
            self.Q = np.diag(np.array(process_var))
        else:
            self.Q = np.eye(3) * process_var

        # Measurement Noise Covariance
        if isinstance(measurement_var, (list, tuple, np.ndarray)):
            self.R = np.diag(np.array(measurement_var))
        else:
            self.R = np.eye(2) * measurement_var

    def initialize(self, init_state):
        """
        Set initial state.
        Args:
            init_state (array-like): [x, y, theta]
        """
        self.x = np.reshape(init_state, (3, 1))

    def set_process_noise(self, process_var):
        """ Update Q after initialization. """
        if isinstance(process_var, (list, tuple, np.ndarray)):
            self.Q = np.diag(np.array(process_var))
        else:
            self.Q = np.eye(3) * process_var

    def set_measurement_noise(self, measurement_var):
        """ Update R after initialization. """
        if isinstance(measurement_var, (list, tuple, np.ndarray)):
            self.R = np.diag(np.array(measurement_var))
        else:
            self.R = np.eye(2) * measurement_var

    def predict(self, control):
        """
        EKF Prediction step.
        Args:
            control (array-like): [v, w] (linear and angular velocities)
        """
        v, w = control
        theta = self.x[2, 0]
        dt = self.dt

        # Nonlinear state prediction
        x_pred = self.x.copy()
        x_pred[0, 0] += v * np.cos(theta) * dt
        x_pred[1, 0] += v * np.sin(theta) * dt
        x_pred[2, 0] += w * dt
        x_pred[2, 0] = (x_pred[2, 0] + np.pi) % (2 * np.pi) - np.pi  # Keep theta in [-pi, pi]

        # Jacobian of the motion model
        F = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0,  1]
        ])

        self.x = x_pred
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        EKF Update step.
        Args:
            z (array-like): Noisy measurement [x_measured, y_measured]
        """
        H = np.array([[1, 0, 0], [0, 1, 0]])
        z = np.reshape(z, (2, 1))
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

    def get_state(self):
        """
        Returns current state estimate as flat array: [x, y, theta]
        """
        return self.x.flatten()
