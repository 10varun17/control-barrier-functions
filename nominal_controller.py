import numpy as np

def control_law(q, q_g):
    return q_g[:2] - q[:2]

def non_linear_transform(delx, dely, x_r, curr_yaw):
    """
    Receives x and y velocity and returns the corresponding linear and angular velocity
    """
    v = (1 / x_r) * (x_r * np.cos(curr_yaw) * delx + x_r * np.sin(curr_yaw) * dely)
    omega = (1 / x_r) * (-1 * np.sin(curr_yaw) * delx + np.cos(curr_yaw) * dely)

    return np.array([v, omega]).reshape(2,1)