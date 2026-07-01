# Franka Bridge

Code to bridge the Octo model and the Franka robot. Different backends are written to experiment and find the best setup.

---

## Bridge Implementations

### JointStates_octo_franka_bridge.py
Runs Octo inference and commands the robot by publishing `sensor_msgs/JointState` to `/gello/joint_states`.

### aio_octo_franka_bridge.py
Uses `aiofranka FrankaRemoteController` with OSC (Operational Space Control) to send end-effector pose targets instead of joint positions. 

---

## Test Scripts
Tests the bridge logic without Octo. Doing so for making sure the ROS2 connections, gripper commands, and joint/eep publishing are done correctly before introducing the model.

### TEST-jointStates.py
Runs Octo inference and commands the robot by publishing `sensor_msgs/JointState` to `/gello/joint_states`.

### TEST-aio.py
Uses `aiofranka FrankaRemoteController` with OSC (Operational Space Control) to send end-effector pose targets instead of joint positions. 

---

## Reference Scripts

### jointStates_practice.py
Moves the robot wrist in a sinusoidal motion by publishing joint states. Using as reference on how to use joint commands.

### gripper_practice.py
Previous code for controlling the gripper and talking to an Arduino over serial. Using as a reference for how to control the gripper. Reads a `Float32` width percent from GELLO, publishes a `Bool` open/closed.

### aio-franka-circle_demo.py
Written by Dr. Dimas Dutra. An example of how to use the `aiofranka` library. Moves the robot to a home position, switches to OSC mode, and runs a circular end-effector trajectory.
