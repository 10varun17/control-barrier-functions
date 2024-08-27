# control-barrier-functions

This repository implements the control barrier functions to avoid a static obstacle while reaching a goal. The virtual agent used in this repo uses unicycle dynamics, which is in the control affine form. Control barrier functions mathematically impose the safety constraints in the robotic system through an optimization problem (quadratic program), which is solved using CasADI.

### Progress
Current implementation allows for the robot to slow down near the obstacle but doesn't control it to steer around the obstacle. This repo is under development.

## References
1. [Formally Correct Composition of Coordinated Behaviors Using Control Barrier Certificates](https://ieeexplore.ieee.org/abstract/document/8594302)
2. [Control Barrier Functions: Theory and Applications](https://ieeexplore.ieee.org/abstract/document/8796030)
3. [Constrained robot control using control barrier functions](https://ieeexplore.ieee.org/abstract/document/7759067)
4. [A Sequential Composition Framework for Coordinating Multi-Robot Behaviors](https://arxiv.org/pdf/1907.07718)
5. [UC Berkeley's Robotics Manipulation and Interaction Course](https://ucb-ee106.github.io/106b-sp23site/)
