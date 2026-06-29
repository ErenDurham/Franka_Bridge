import itertools
import time

import numpy as np
import aiofranka
from aiofranka import FrankaRemoteController

T = 2.5
freq = 50
r = 0.05

dt = 1.0 / freq

# Unlock the robot (opens brakes + activates FCI)
# aiofranka.unlock()

# Create controller and start server subprocess
controller = FrankaRemoteController()
controller.start()

# Use the robot
controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])

controller.switch("osc")

# Task-space gains [x, y, z, roll, pitch, yaw]
controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])
controller.ee_kd = np.ones(6) * 10.0

# Null-space gains (keeps robot away from joint limits)
controller.null_kp = np.ones(7) * 10.0
controller.null_kd = np.ones(7) * 1.0

x0, y0, z0 = controller.initial_ee[:3, 3]
ee_desired = controller.initial_ee.copy()

for i in itertools.count():
    theta = i * 2 * np.pi * dt / T
    state = controller.state
    ee_desired[0, 3] = x0
    ee_desired[1, 3] = y0 + r * np.sin(theta)
    ee_desired[2, 3] = z0 + r * np.cos(theta) - r
    controller.set("ee_desired", ee_desired)

# Stop server
controller.stop()
