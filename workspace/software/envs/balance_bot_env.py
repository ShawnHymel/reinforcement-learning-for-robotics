"""
Gymnasium wrapper for MuJoCo simulator that loads and runs the environment for a simple 2-wheel
balance bot.

Author: Shawn Hymel
Date: April 28, 2026
"""

# Standard libraries
import math
from pathlib import Path

# Third-party libraries
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class BalanceBotEnv(gym.Env):
    """
    Define our own gymnasium environment class that wraps the MuJoCo simulator so we can interact
    with it using the standard gymnasium methods (e.g. step(), reset(), action_space, etc.).
    """

    # Attribute: declare which render modes are available and set the target FPS for rendering
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        mjcf_path="balance_bot.xml",
        max_steps=10000,
        render_mode=None,
        alpha=0.99,
        sensor_imu_accel="imu_accel",
        sensor_imu_gyro="imu_gyro",
        sensor_left_wheel_vel="left_wheel_vel",
        sensor_right_wheel_vel="right_wheel_vel",
        actuator_left_motor="left_motor",
        actuator_right_motor="right_motor",
        alive_bonus=1.0, 
        pitch_penalty_coef=5.0, 
        action_penalty_coef=0.01,
        position_penalty_coef=0.01,
        yaw_penalty_coef=0.1,
        tip_threshold_deg=30.0,
    ):
        """
        Constructor: Initialize the balance bot environment

        Args:
            mjcf_path (str or Path): Path to the MuJoCo MJCF model file (XML)
            max_steps (int): Max number of simulation steps per episode before resetting
            render_mode (str or None): Visual output, one of [None, "human", "rgb_array"]
            alpha (float): Complementary filter coefficient, higher = more gyro, lower = more accel
            sensor_imu_accel (str): MJCF name of the IMU accelerometer sensor
            sensor_imu_gyro (str): MJCF name of the IMU gyroscope sensor
            sensor_left_wheel_vel (str): MJCF name of the left wheel velocity sensor
            sensor_right_wheel_vel (str): MJCF name of the right wheel velocity sensor
            actuator_left_motor (str): MJCF name of the left motor actuator
            actuator_right_motor (str): MJCF name of the right motor actuator
            alive_bonus (float): Reward given each step the robot stays upright,
            pitch_penalty_coef (float): Scales the pitch^2 penalty, encourage staying upright
            action_penalty_coef (float): Scales the action^2 penalty, discourage jittery motion
            position_penalty_coef (float): Scales the position penalty (x^2 + y^2), discourages
                                           drifting from the starting position
            yaw_penalty_coef (float): Scales the abs(yaw_rate) penalty, discourages spinning around
                                      the Z axis
            tip_threshold_deg (float): Angle (degrees) in which the robot is considered tipped
        """
        # Load model into MuJoCo and get simulation state
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)

        # Set render mode
        self.render_mode = render_mode

        # Store complementary filter coefficient
        self.alpha = alpha

        # Store sensor names
        self.sensor_imu_accel = sensor_imu_accel
        self.sensor_imu_gyro = sensor_imu_gyro
        self.sensor_left_wheel_vel = sensor_left_wheel_vel
        self.sensor_right_wheel_vel = sensor_right_wheel_vel

        # Get ID of actuators from MJCF names
        self.left_motor_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR, 
            actuator_left_motor
        )
        self.right_motor_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_right_motor
        )

        # Define observation space (i.e. what the agent can see) and limits
        # [pitch, pitch_rate, wheel_vel_left, wheel_vel_right]
        obs_low  = np.array([-np.pi, -20.0, -50.0, -50.0], dtype=np.float32)
        obs_high = np.array([ np.pi,  20.0,  50.0,  50.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Define action space (i.e. what the agent can do) and limits, normalized to [-1, 1]
        # [left_wheel_torque, right_wheel_torque]
        actions_low = np.array([-1.0, -1.0], dtype=np.float32)
        actions_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

        # Save reward coefficients
        self.alive_bonus = alive_bonus
        self.pitch_penalty_coef = pitch_penalty_coef
        self.action_penalty_coef = action_penalty_coef
        self.position_penalty_coef = position_penalty_coef
        self.yaw_penalty_coef = yaw_penalty_coef

        # Save tip threshold
        self.tip_threshold_deg = tip_threshold_deg

        # Number of steps to take before resetting the episode
        self.max_steps = max_steps

        # Set initial pitch state
        self._pitch = 0.0

        # Initialize the viewer
        self._viewer = None

        # Internal step counter
        self._step = 0

    def _get_obs(self):
        """
        Read sensor data from MuJoCo and return the observation vector.

        Returns:
            np.ndarray: [pitch, pitch_rate, wheel_vel_left, wheel_vel_right]
        """
        # Read raw IMU sensor data
        accel_x, _, accel_z = self.data.sensor(self.sensor_imu_accel).data
        pitch_rate = self.data.sensor(self.sensor_imu_gyro).data[1]

        # Accelerometer-derived pitch estimate
        accel_pitch = -1 * math.atan2(accel_x, accel_z)

        # Copmlementary filter to estimate pitch from accelerometer and gyroscope
        self._pitch = self.alpha * (self._pitch + pitch_rate * self.model.opt.timestep) + \
                    (1 - self.alpha) * accel_pitch

        # Read wheel velocities from sensors
        wheel_vel_left  = self.data.sensor(self.sensor_left_wheel_vel).data[0]
        wheel_vel_right = self.data.sensor(self.sensor_right_wheel_vel).data[0]

        return np.array([self._pitch, pitch_rate, wheel_vel_left, wheel_vel_right], \
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment and bot to an initial state.

        Args:
            seed (int or None): seed for the random number generator(s)
            options (dict or None): Unused, required by the Gymnasium API

        Returns:
            observation (np.ndarray): Initial observation
            info (dict): Empty, no extra info returned on reset
        """
        super().reset(seed=seed)

        # Reset the simulator
        mujoco.mj_resetData(self.model, self.data)

        # Reset pitch
        self._pitch = 0.0

        # Impart an initial angular velocity around the y axis so the agent learns to recover
        # Note: qvel[4] = wy (rad/s)
        self.data.qvel[4] += self.np_random.uniform(-0.5, 0.5)

        # Update the state of the robot without taking a full time step
        mujoco.mj_forward(self.model, self.data)

        # Reset the step counter
        self._step = 0

        return self._get_obs(), {}

    def step(self, action):
        """
        Advance the simulation by one step and return the result.

        Args:
            action (np.ndarray): [left_wheel_torque, right_wheel_torque], normalized to [-1, 1]

        Returns:
            obs (np.ndarray): Observation from _get_obs()
            reward (float): Reward signal for this step
            terminated (bool): True if the robot has tipped/fallen over
            truncated (bool): True if the episode has reached max_steps
            info (dict): Empty, no extra info returned
        """
        # Set motors to given (normalized) torque
        self.data.ctrl[self.left_motor_id]  = action[0]
        self.data.ctrl[self.right_motor_id] = action[1]

        # Advance simulation by one step
        mujoco.mj_step(self.model, self.data)
        self._step += 1

        # Get observation
        obs = self._get_obs()
        pitch = obs[0]

        # Reward function: alive - (A*pitch^2) - (B*action^2) - (C*(x^2 + y^2)) - D*abs(yaw)
        # Note: qpos (simulation state) only available during training
        #   alive: reward for staying upright each step
        #   pitch: penalty for leaning
        #   action: penalty for jittery motor commands
        #   position: penalty for drifting from the starting position
        #   yaw: penalty for rotating around Z axis
        pitch_penalty = self.pitch_penalty_coef * pitch**2
        action_penalty = self.action_penalty_coef * np.sum(action**2)
        x_pos = self.data.qpos[0]
        y_pos = self.data.qpos[1]
        position_penalty = self.position_penalty_coef * (x_pos**2 + y_pos**2)
        yaw_rate = self.data.qvel[5]
        yaw_penalty = self.yaw_penalty_coef * abs(yaw_rate)
        reward = self.alive_bonus - pitch_penalty - action_penalty - position_penalty - yaw_penalty

        # Termination (if robot tips or we run out of time in the episode)
        terminated = abs(pitch) > math.radians(self.tip_threshold_deg)
        truncated = self._step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def render(self):
        """
        Render the current simulation state to the MuJoCo viewer window.
        """
        if self.render_mode != "human":
            return

        # Create the viewer on the first render call
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

            # Set up the camera
            self._viewer.cam.type     = mujoco.mjtCamera.mjCAMERA_FREE
            self._viewer.cam.lookat[:] = [0, 0, 0.05]
            self._viewer.cam.distance  = 0.8
            self._viewer.cam.azimuth   = 45
            self._viewer.cam.elevation = -25

        # Push the current simulation state to the viewer
        self._viewer.sync()

    def close(self):
        """
        Clean up the viewer and simulation resources.
        Automatically called by Gymnasium when the environment is done.
        """
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None