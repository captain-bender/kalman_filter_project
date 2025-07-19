import numpy as np
from ekf_filter import EKFDD
from kalman_filter import KalmanFilterCV

def test_kalmanfiltercv_initialization():
    kf = KalmanFilterCV(dt=1.0, process_var=0.01, measurement_var=1.0)
    assert kf.x.shape == (4, 1)
    assert np.allclose(kf.P, np.eye(4) * 500)

def test_kalmanfiltercv_predict_update():
    kf = KalmanFilterCV(dt=1.0, process_var=0.0, measurement_var=1e-8)
    kf.initialize([0,0], [1,1])
    z = np.array([1.0, 1.0])  # After 1 timestep, position should be [1,1]
    kf.predict()
    kf.update(z)
    est = kf.get_state()
    assert np.allclose(est[:2], [1, 1], atol=1e-5)

def test_ekfdd_initialization():
    ekf = EKFDD(dt=1.0, process_var=[0.01, 0.01, 0.001], measurement_var=[0.25, 0.5])
    ekf.initialize([0,0,0])
    assert ekf.x.shape == (3,1)
    assert np.allclose(np.diag(ekf.Q), [0.01, 0.01, 0.001])

def test_ekfdd_predict_update():
    ekf = EKFDD(dt=1.0, process_var=[0,0,0], measurement_var=[1e-8,1e-8])
    # No noise: after move and update, should perfectly match
    ekf.initialize([0, 0, 0])
    v, w = 1.0, 0.0
    ekf.predict([v, w])
    # Should move to (1, 0, 0)
    ekf.update([1.0, 0.0])
    state = ekf.get_state()
    assert np.allclose(state, [1.0, 0.0, 0.0], atol=1e-5)
