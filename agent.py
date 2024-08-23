import numpy as np

class Agent:
    def __init__(self, x0, y0, yaw0):
        self._q0 = np.array([x0, y0, yaw0]).reshape(3, 1)
        self._x_next = x0
        self._y_next = y0
        self._yaw_next = yaw0
        self._q = self._q0
        self._dt = 0.001

    def get_state(self):
        return self._q
    
    def update_state(self, u):
        v = u[0, 0]
        omega = u[1, 0]

        self._x_next += self._dt * (v * np.cos(self._yaw_next))
        self._y_next += self._dt * (v * np.sin(self._yaw_next))
        self._yaw_next += self._dt * omega

        self._q[0][0] = self._x_next
        self._q[1][0] = self._y_next
        self._q[2][0] = self._yaw_next
