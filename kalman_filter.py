import numpy as np

class KalmanFilterCV:
    def __init__(self, dt, process_var, measurement_var):
        """
        Kalman filter for a linear constant-velocity motion model in 2D.
        
        Args:
            dt (float): Time step (seconds)
            process_var (float): Process (model) noise variance
            measurement_var (float): Measurement noise variance
        """
        self.dt = dt

        # State vector: [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        
        # State transition matrix (moves position and velocity forward by dt)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we measure position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariance matrices
        self.P = np.eye(4) * 500  # Initial estimate uncertainty
        self.Q = np.eye(4) * process_var  # Process noise covariance
        self.R = np.eye(2) * measurement_var  # Measurement noise covariance

    def initialize(self, init_pos, init_vel=(0, 0)):
        self.x = np.array([[init_pos[0]], [init_pos[1]], [init_vel[0]], [init_vel[1]]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        z: observed position, array-like, shape (2,)
        """
        z = np.reshape(z, (2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_state(self):
        return self.x.flatten()
