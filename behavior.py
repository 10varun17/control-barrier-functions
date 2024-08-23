import numpy as np
import agent
import cbf_utils
import obstacle
import time
import nominal_controller
import csv

# init the goal, agent, and obstacle
q_goal = np.array([8.0, 4.0, 0.0]).reshape(3, 1)
robot = agent.Agent(1.0, 4.0, -3.14)
obs = obstacle.CircleObstacle(4.0, 4.0, 0.5)

# init the tolerance, max linear vel, and max angular vel
tol = 0.01
v_max = 0.4
omega_max = 0.523

t_0 = time.time()
dist_to_goal = np.linalg.norm(robot.get_state()[:2] - q_goal[:2])

csv_file = "data.csv"
headers = ["x", "y"]
with open(csv_file, "w") as outfile:
    csv_writer = csv.DictWriter(outfile, fieldnames=headers)
    csv_writer.writeheader()

while dist_to_goal > 0.1:
    q_curr = robot.get_state()
    # Nominal control input
    dq = nominal_controller.control_law(q_curr, q_goal)
    dx = dq[0][0]
    dy = dq[1][0]
    u_nom = nominal_controller.non_linear_transform(dx, dy, 0.1, q_curr[2][0])

    # Limit the control input within the max value
    v, omega = u_nom[0][0], u_nom[1][0]
    v = min(max(-v_max, v), v_max)
    omega = min(max(-omega_max, omega), omega_max)
    u_nom[0][0] = v
    u_nom[1][0] = omega

    # ### optimize the control input to keep the robot within the safe set
    # get the control barrier function
    h = cbf_utils.compute_h(q_curr, obs)

    # compute the lie derivatives
    lf_h = cbf_utils.compute_lfh(q_curr, obs)
    lg_h = cbf_utils.compute_lgh(q_curr, obs)

    # optimize the control input
    u_safe = cbf_utils.cbf_qp_controller(u_nom, h, lf_h, lg_h)

    # send the control input to the robot
    robot.update_state(u_safe)

    # write the data to csv file
    with open(csv_file, "a") as outfile:
        csv_writer = csv.DictWriter(outfile, fieldnames=headers)

        data = {
            "x": q_curr[0][0],
            "y": q_curr[1][0]
        }

        csv_writer.writerow(data)
    
    dist_to_goal = np.linalg.norm(robot.get_state()[:2] - q_goal[:2]) 

