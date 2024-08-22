import numpy as np

class Agent:
    def __init__(self, x0, y0, phi0):
        self._q0 = np.array([x0, y0, phi0]).reshape(3, 1)
        self._q = self._q0

    def get_state(self):
        return self._q
    
    def update_state(self, u):
        pass