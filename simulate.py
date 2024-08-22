import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.patches import Circle
import pandas as pd

fig, ax = plt.subplots()

def animate(i):
    data = pd.read_csv("data.csv")
    x = data["x"]
    y = data["y"]
    plt.cla()

    # Plot the obstacle
    circle = Circle((4, 4.), 0.5, fill=False)
    ax.add_patch(circle)
    plt.scatter(4., 4., c="red")

    # Plot the goal
    plt.scatter(8., 4, c="green", s = 90)
    plt.scatter(x, y, c="blue", s=45)
    plt.title("Trajectory of the robot")
    plt.tight_layout()
    plt.axis([0, 10, 0, 5])

ani = FuncAnimation(fig, animate, interval=1000)
plt.tight_layout()
plt.show()