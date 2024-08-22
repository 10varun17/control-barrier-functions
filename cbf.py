import numpy as np
import casadi as ca

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

def nominal_controller(delx, dely, x_r, curr_yaw):
    """
    Receives x and y velocity and returns the corresponding linear and angular velocity
    """
    v = (1 / x_r) * (x_r * np.cos(curr_yaw) * delx + x_r * np.sin(curr_yaw) * dely)
    omega = (1 / x_r) * (-1 * np.sin(curr_yaw) * delx + np.cos(curr_yaw) * dely)

    return np.array([v, omega]).reshape(2,1)

def compute_h(q, obs : CircleObstacle):
    xo, yo, ro = obs.get_x(), obs.get_y(), obs.get_radius()
    x, y = q[0, 0], q[1, 0]
    return (x - xo)**2 + (y - yo)**2 - ro**2

def compute_hdot(q, qdot, obs: CircleObstacle):
    qo = np.array([obs.get_x(), obs.get_y(), obs.get_radius()]).reshape(3,1)
    return 2 * (q[:2, 0] - qo[:2, 0]).T @ qdot[:2, 0]

def compute_lfh(q, obs):
    xo, yo, ro = obs.get_x(), obs.get_y(), obs.get_radius()
    x, y, phi = q[0, 0], q[1, 0], q[2, 0]
    h = (x - xo)**2 + (y - yo)**2 - ro**2
    dh_dq = np.array([
        [2*(x - xo)],
        [2*(y - yo)],
        [0]
    ])
    f_q = f(q)
    return dh_dq[:, 0].T @ f_q[:,0]

def compute_lgh(q, obs):
    xo, yo, ro = obs.get_x(), obs.get_y(), obs.get_radius()
    x, y, phi = q[0, 0], q[1, 0], q[2, 0]
    h = (x - xo)**2 + (y - yo)**2 - ro**2

    dh_dq = np.array([
        [2*(x - xo)],
        [2*(y - yo)],
        [0]
    ])
    g_q = g(q)
    g1_q = g_q[:, 0]
    g2_q = g_q[:, 1]

    lg_h = np.zeros((2,))
    lg_h[0] = dh_dq[:, 0].T @ g1_q
    lg_h[1] = dh_dq[:, 0].T @ g2_q

    return lg_h

def cbf_qp_controller(u_nom, h, hdot, gamma=1.0):
    opti = ca.Opti()
    u_safe = opti.variable(2,1)
    opti.subject_to(hdot + gamma*h >= 0)
    opti.set_initial(u_safe, u_nom)
    cost_func = (u_safe - u_nom).T @ (u_safe - u_nom)
    opti.minimize(cost_func)
    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt", option)
    sol = opti.solve()
    u_safe_opti = sol.value(u_safe)
    return u_safe_opti


