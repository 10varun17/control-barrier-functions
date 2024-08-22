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