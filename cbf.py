import numpy as np
from jax import grad

class CircleObstacle:
    def __init__(self, x, y, r):
        self._x = x
        self._y = y
        self._radius = r

    def get_x(self):
        return self._x
    
    def get_y(self):
        return self._y
    
    def get_radius(self):
        return self._radius

def f(q):
    return np.zeros((3,1))

def g(q):
    phi = q[2, 0]
    g_q = np.array([
        [np.cos(phi), 0],
        [np.sin(phi), 0],
        [0, 1]
    ])
    return g_q

def control_affine_dynamics(q, u):
    """
    Creates the control affine dynamics for single integrator.
    """
    return f(q) + g(q) @ u

def control_law(q, q_g):
    return q_g[:2] - q[:2]

def non_linear_transform(delx, dely, x_r, curr_yaw):
    """
    Receives x and y velocity and returns the corresponding linear and angular velocity
    """
    v = (1 / x_r) * (x_r * np.cos(curr_yaw) * delx + x_r * np.sin(curr_yaw) * dely)
    omega = (1 / x_r) * (-1 * np.sin(curr_yaw) * delx + np.cos(curr_yaw) * dely)

    return v, omega

def compute_h(q, obs : CircleObstacle):
    xo, yo, ro = obs.get_x(), obs.get_y(), obs.get_radius()
    x, y = q[0, 0], q[1, 0]
    return (x - xo)**2 + (y - yo)**2 - ro**2

def compute_hdot(q, qdot, obs: CircleObstacle):
    qo = np.array([obs.get_x(), obs.get_y(), obs.get_radius()]).reshape(3,1)
    return 2 * (q[:2, 0] - qo[:2, 0]).T @ qdot[:2, 0]

def cbf_safe_dist(x_diff: np.ndarray, d_safe):
    """
    Returns the control barrier function for maintaining safe distance

    Args:
        x_diff: the relative state of two agents (x_diff = x_i - x_j)
        d_safe: the safety distance between agents i and j
    """
    return -x_diff.T @ x_diff + d_safe**2 

def grad_f(f, params, argnums=0):
    """
    Computes the gradient of f with respect to variable at argnums in f and evaluetes at (x_diff, d_safe)
    """
    params_tuple = tuple(params)
    grad_f = grad(f, argnums)
    grad_value = grad_f(*params_tuple)
    return np.array(grad_value)

def lie_derivative(h_func, f_func, params, argnums=0):
    grad_h = grad_f(h_func, params, argnums)
    f_x = f_func(params[argnums])
    return grad_h.T @ f_x

# x_agent = np.array([1.0, -2.0])
# u_agent = np.array([-0.4, 0.2])

# x_obs = np.array([2.0, 3.0])
# u_obs = np.array([-0.3, 0.1])

# x_diff = x_agent - x_obs
# x_dot = control_affine_dynamics(x_agent, u_agent) - control_affine_dynamics(x_obs, u_obs) # difference of the dynamics.. for 2d si, it's u_agent - u_obs

# agent_rad = 0.5
# h = cbf_safe_dist(x_diff, agent_rad) # control barrier function h = norm**2 - dist**2

# grad_h = grad_f(cbf_safe_dist, params=[x_diff, agent_rad])
# lg_h = grad_h.T@x_dot

# print(lie_derivative(cbf_safe_dist, f, params=[x_diff, agent_rad]))

