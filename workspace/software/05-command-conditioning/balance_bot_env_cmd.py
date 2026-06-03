"""
Gymnasium wrapper for MuJoCo simulator that loads and runs the environment for a simple 2-wheel
balance bot. Adds command-conditioned rewards on top of domain randomization.

Author: Shawn Hymel
Date: May 20, 2026
"""

# Standard libraries
from dataclasses import dataclass
import math
from pathlib import Path

# Third-party libraries
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


@dataclass  
class DomainRandomConfig:
    """
    Configuration for domain randomization in BalanceBotEnv. Pass an instance to BalanceBotEnv's
    domain_rand parameter to enable.  All values default to disabled (0.0, False, or identity 
    ranges).
    
    Attributes:
        pitch_noise_std_dev: Standard deviation of Gaussian noise added to pitch observation. 
                             Simulates IMU noise.
        pitch_rate_noise_std_dev: Standard deviation of Gaussian noise added to pitch rate
                                  observation. Simulates IMU noise.
        wheel_vel_noise_std_dev:  Standard deviation of Gaussian noise added to wheel velocity
                                  observation. Simulates encoder noise.
        action_delay_steps: Number of steps to delay actions (0=disabled). Simulates I2C and motor 
                            response latency.
        action_delay_random: If True, randomize delay 0..action_delay_steps each episode.
        motor_noise_scale: Uniform noise magnitude added to motor commands. Simulates motor driver 
                           noise and tire ridge impulses.
        push_prob: Probability per step of applying a random external force. Simulates bumps, 
                   nudges, and uneven terrain effects.
        push_force_max_n: Maximum magnitude of random push force in Newtons.
        mass_scale_range: (min, max) scaling factor for chassis mass each episode. Simulates payload
                          variation and model uncertainty.
        friction_scale_range: (min, max) scaling factor for wheel-to-ground friction each episode.
                              Simulates different floor surfaces.
        motor_gain_range: (min, max) Simulate motor torque variance (e.g. battery sag)
        ridge_prob: Probability of applying a random torque to the wheel axles to simulate the tire
                    ridges hitting the ground
        ridge_torque_max_nm: Max random torque to apply to axles (N-m)
        cmd_vel_range: (min, max) normalized velocity command, e.g. (-1.0, 1.0)
        cmd_yaw_range: (min, max) normalized yaw rate command, e.g. (-1.0, 1.0)
        cmd_zero_prob: probability of sampling vel/yaw=0 (forces some stillness)
        cmd_resample_prob: probability of sampling a new command each step
    """
    pitch_noise_std_dev: float = 0.0
    pitch_rate_noise_std_dev: float = 0.0
    wheel_vel_noise_std_dev: float = 0.0
    action_delay_steps: int = 0
    action_delay_random: bool = False
    motor_noise_scale: float = 0.0
    push_prob: float = 0.00
    push_force_max_n: float = 0.0
    mass_scale_range: tuple = (1.0, 1.0)
    friction_scale_range: tuple = (1.0, 1.0)
    motor_gain_range: tuple = (1.0, 1.0)
    ridge_prob: float = 0.0
    ridge_torque_max_nm: float = 0.0
    cmd_vel_range: tuple = (0.0, 0.0)
    cmd_yaw_range: tuple = (0.0, 0.0)           
    cmd_zero_prob: float = 0.0
    cmd_resample_prob: float = 0.0
    

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
        debug=False,
        max_steps=10000,
        render_mode=None,
        alpha=0.99,
        sensor_imu_accel="imu_accel",
        sensor_imu_gyro="imu_gyro",
        sensor_left_wheel_vel="left_wheel_vel",
        sensor_right_wheel_vel="right_wheel_vel",
        actuator_left_motor="left_motor",
        actuator_right_motor="right_motor",
        left_wheel_joint="left_wheel_joint",
        right_wheel_joint="right_wheel_joint",
        alive_bonus=1.0, 
        pitch_penalty_coef=5.0, 
        pitch_rate_penalty_coef=0.0,
        action_penalty_coef=0.01,
        position_penalty_coef=0.01,
        smoothness_penalty_coef=0.0,
        vel_reward_coef=1.0,
        yaw_reward_coef=1.0,
        sigma_vel=0.5,
        sigma_yaw=0.5,
        vel_max=1.0,
        yaw_max=2.0,
        tip_threshold_deg=30.0,
        domain_rand=None,
    ):
        """
        Constructor: Initialize the balance bot environment

        Args:
            mjcf_path (str or Path): Path to the MuJoCo MJCF model file (XML)
            debug (bool): Whether or not to print out debugging info (e.g. velocity targets)
            max_steps (int): Max number of simulation steps per episode before resetting
            render_mode (str or None): Visual output, one of [None, "human", "rgb_array"]
            alpha (float): Complementary filter coefficient, higher = more gyro, lower = more accel
            sensor_imu_accel (str): MJCF name of the IMU accelerometer sensor
            sensor_imu_gyro (str): MJCF name of the IMU gyroscope sensor
            sensor_left_wheel_vel (str): MJCF name of the left wheel velocity sensor
            sensor_right_wheel_vel (str): MJCF name of the right wheel velocity sensor
            actuator_left_motor (str): MJCF name of the left motor actuator
            actuator_right_motor (str): MJCF name of the right motor actuator
            left_wheel_joint (str): MJCF name of the left wheel joint
            right_wheel_joint (str): MJCF name of the right wheel joint
            alive_bonus (float): Reward given each step the robot stays upright,
            pitch_penalty_coef (float): Scales the pitch^2 penalty, encourage staying upright
            pitch_rate_penalty_coef (float): Scales the pitch rate penalty, discourage rapid changes
            action_penalty_coef (float): Scales the action^2 penalty, discourage jittery motion
            position_penalty_coef (float): Scales the position penalty (x^2 + y^2), discourages
                                drifting from the starting position
            smoothness_penalty_coef (float): Scales the penalty for changing motor directions
            vel_reward_coef (flaot): Scales the reward for velocity tracking (how close actual 
                                     velocity is to target velocity)
            yaw_reward_coef (float): Scales the reward for yaw tracking (how close actual turn/yaw 
                                     rate is to target rate)
            sigma_vel (float): Controls velocity tracking tolerance in the exponential reward. 
                               Smaller values mean less reward when there's an error.
            sigma_yaw (float): Controls yaw tracking tolerance in the exponential reward. Smaller
                               values mean less reward when there's an error.
            vel_max (float): Max forward/back velocity in m/s.
            yaw_max (float): Max yaw rate (rad/s), scales normalized cmd_yaw to actual units
            tip_threshold_deg (float): Angle (degrees) in which the robot is considered tipped
            domain_rand (DomainRandomConfig): Configuration for performing various domain
                                              randomizations (None to disable)
        """
        # Set debug level
        self._debug = debug

        # Load model into MuJoCo and get simulation state
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)

        # Get handle to ground geometry (must match name in MJCF file)
        self._ground_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "ground",
        )
        assert self._ground_id  != -1, "Geom 'ground' not found in MJCF"

        # Get handle to ground geometry (must match name in MJCF file)
        self._chassis_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "chassis",
        )
        assert self._chassis_id != -1, "Body 'chassis' not found in MJCF"

        # Get DOF index to left wheel joint
        left_wheel_joint_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, left_wheel_joint
        )
        assert left_wheel_joint_id !=-1, "Left wheel joint not found in MJCF"
        self._left_wheel_dof_idx = self.model.jnt_dofadr[left_wheel_joint_id]

        # Get DOF index to right wheel joint
        right_wheel_joint_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, right_wheel_joint
        )
        assert right_wheel_joint_id !=-1, "Right wheel joint not found in MJCF"
        self._right_wheel_dof_idx = self.model.jnt_dofadr[right_wheel_joint_id]

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

        # Save original mass, friction, and gearing values
        self._chassis_mass_orig = float(self.model.body_mass[self._chassis_id])
        self._ground_friction_orig = float(self.model.geom_friction[self._ground_id, 0])
        self._left_gear_orig = float(self.model.actuator_gear[self.left_motor_id, 0])
        self._right_gear_orig = float(self.model.actuator_gear[self.right_motor_id, 0])

        # Define observation space (i.e. what the agent can see) and limits
        # [pitch, pitch_rate, wheel_vel_left, wheel_vel_right, cmd_vel, cmd_yaw]
        obs_low  = np.array([-np.pi, -20.0, -50.0, -50.0, -1.0, -1.0], dtype=np.float32)
        obs_high = np.array([ np.pi,  20.0,  50.0,  50.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Define action space (i.e. what the agent can do) and limits, normalized to [-1, 1]
        # [left_wheel_torque, right_wheel_torque]
        actions_low = np.array([-1.0, -1.0], dtype=np.float32)
        actions_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

        # Save reward coefficients
        self.alive_bonus = alive_bonus
        self.pitch_penalty_coef = pitch_penalty_coef
        self.pitch_rate_penalty_coef = pitch_rate_penalty_coef
        self.action_penalty_coef = action_penalty_coef
        self.position_penalty_coef = position_penalty_coef
        self.smoothness_penalty_coef = smoothness_penalty_coef
        self.vel_reward_coef = vel_reward_coef
        self.yaw_reward_coef = yaw_reward_coef

        # Save the sigmas and check that they are greater than 0
        self.sigma_vel = sigma_vel
        if self.sigma_vel <= 0.0:
            print("Warning: sigma_vel is {sigma_vel}, which must be > 0. Setting to 1e-6.")
            self.sigma_vel = 1e-6
        self.sigma_yaw = sigma_yaw
        if self.sigma_yaw <= 0.0:
            print("Warning: sigma_yaw is {sigma_yaw}, which must be > 0. Setting to 1e-6.")
            self.sigma_yaw = 1e-6

        # Save max velocity values
        self.vel_max = vel_max
        self.yaw_max = yaw_max

        # Save tip threshold
        self.tip_threshold_deg = tip_threshold_deg

        # Number of steps to take before resetting the episode
        self.max_steps = max_steps

        # Save domain randomization
        self.dr = domain_rand

        # Action delay buffer
        self._action_delay = 0
        self._action_buffer = []

        # Set initial pitch state
        self._pitch = 0.0

        # Set current command state
        self._cmd_vel = 0.0
        self._cmd_yaw = 0.0
        self._cmd_zero_pos = np.array([0.0, 0.0]) 

        # Initialize previous action (used for smoothness penalty)
        self._prev_action = np.zeros(2, dtype=np.float32)

        # Initialize the viewer
        self._viewer = None

        # Internal step counter
        self._step = 0

    def _get_obs(self):
        """
        Read sensor data from MuJoCo and return the observation vector.

        Returns:
            np.ndarray: [pitch, pitch_rate, wheel_vel_left, wheel_vel_right, cmd_vel, cmd_yaw]
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

        # Construct observation
        obs = np.array([self._pitch, pitch_rate, wheel_vel_left, wheel_vel_right,
                self._cmd_vel, self._cmd_yaw], dtype=np.float32)

        # Optionally add Gaussian noise to observations
        if self.dr is not None:
            if self.dr.pitch_noise_std_dev > 0.0:
                obs[0] += self.np_random.normal(0.0, self.dr.pitch_noise_std_dev)
            if self.dr.pitch_rate_noise_std_dev > 0.0:
                obs[1] += self.np_random.normal(0.0, self.dr.pitch_rate_noise_std_dev)
            if self.dr.wheel_vel_noise_std_dev > 0.0:
                obs[2] += self.np_random.normal(0.0, self.dr.wheel_vel_noise_std_dev)
                obs[3] += self.np_random.normal(0.0, self.dr.wheel_vel_noise_std_dev)

        return obs

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

        # Optionally randomize the mass
        if self.dr is not None and self.dr.mass_scale_range != (1.0, 1.0):
            scale = self.np_random.uniform(
                self.dr.mass_scale_range[0],
                self.dr.mass_scale_range[1],
            )
            self.model.body_mass[self._chassis_id] = scale * self._chassis_mass_orig

        # Optionally randomize friction
        if self.dr is not None and self.dr.friction_scale_range != (1.0, 1.0):
            scale = self.np_random.uniform(
                self.dr.friction_scale_range[0],
                self.dr.friction_scale_range[1],
            )
            self.model.geom_friction[self._ground_id, 0] = scale * self._ground_friction_orig

        # Optionally randomize motor max torque (by randomly scaling the gearing)
        if self.dr is not None and self.dr.motor_gain_range != (1.0, 1.0):
            scale = self.np_random.uniform(
                self.dr.motor_gain_range[0],
                self.dr.motor_gain_range[1],
            )
            self.model.actuator_gear[self.left_motor_id, 0] = scale * self._left_gear_orig
            self.model.actuator_gear[self.right_motor_id, 0] = scale * self._right_gear_orig

        # Optionally sample a new command for the episode (set to zero if DR not used)
        self._cmd_vel = 0.0
        self._cmd_yaw = 0.0
        if self.dr is not None:
            if self.np_random.random() >= self.dr.cmd_zero_prob:
                self._cmd_vel = float(self.np_random.uniform(*self.dr.cmd_vel_range))
                self._cmd_yaw = float(self.np_random.uniform(*self.dr.cmd_yaw_range))
            if self._debug:
                print(f"New episode: cmd_vel={self._cmd_vel:.3f}, cmd_yaw={self._cmd_yaw:.3f}")

        # Clear any applied forces
        self.data.xfrc_applied[self._chassis_id, :] = 0.0
        self.data.qfrc_applied[self._left_wheel_dof_idx]  = 0.0
        self.data.qfrc_applied[self._right_wheel_dof_idx] = 0.0

        # Reset the simulator
        mujoco.mj_resetData(self.model, self.data)

        # Reset pitch
        self._pitch = 0.0

        # Initialize previous action (used for smoothness penalty)
        self._prev_action = np.zeros(2, dtype=np.float32)

        # Impart an initial angular velocity around the y axis so the agent learns to recover
        # Note: qvel[4] = wy (rad/s)
        self.data.qvel[4] += self.np_random.uniform(-0.5, 0.5)

        # Update the state of the robot without taking a full time step
        mujoco.mj_forward(self.model, self.data)

        # Save the robot's x, y location
        self._cmd_zero_pos = np.array([self.data.qpos[0], self.data.qpos[1]])

        # Reset the step counter
        self._step = 0

        # Reset the action delay
        if self.dr is not None:
            # Randomize action delay for this episode
            if self.dr.action_delay_steps > 0 and self.dr.action_delay_random:
                self._action_delay = self.np_random.integers(
                    0, self.dr.action_delay_steps + 1
                )
            else:
                self._action_delay = self.dr.action_delay_steps

            # Fill action buffer with zeros
            self._action_buffer = []
            for _ in range(self._action_delay):
                self._action_buffer.append(np.zeros(self.action_space.shape))

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
        # Apply action delay: add current action to buffer and pop off previously added action
        if self.dr is not None and self._action_delay > 0:
            self._action_buffer.append(action.copy())
            action = self._action_buffer.pop(0)

        # Add random perturbation (noise) to motor commands
        if self.dr is not None and self.dr.motor_noise_scale > 0.0:
            noise = self.np_random.uniform(
                -self.dr.motor_noise_scale,
                self.dr.motor_noise_scale,
                size=action.shape
            )
            action = np.clip(action + noise, -1.0, 1.0)

        # Set motors to given (normalized) torque
        self.data.ctrl[self.left_motor_id]  = action[0]
        self.data.ctrl[self.right_motor_id] = action[1]

        # Clear any applied forces before randomly adding them (if enabled)
        self.data.xfrc_applied[self._chassis_id, :] = 0.0
        self.data.qfrc_applied[self._left_wheel_dof_idx]  = 0.0
        self.data.qfrc_applied[self._right_wheel_dof_idx] = 0.0

        # Apply random external force (push) to chassis in X and Y directions
        if self.dr is not None and self.dr.push_prob > 0.0:
            if self.np_random.random() < self.dr.push_prob:
                # Get random force in X and Y directions
                push_x = self.np_random.uniform(
                    -self.dr.push_force_max_n,
                    self.dr.push_force_max_n,
                )
                push_y = self.np_random.uniform(
                    -self.dr.push_force_max_n,
                    self.dr.push_force_max_n
                )

                # Apply push to chassis
                self.data.xfrc_applied[self._chassis_id, 0] = push_x
                self.data.xfrc_applied[self._chassis_id, 1] = push_y

        # Apply random torque to the axles to simulate the tire ridges hitting the ground
        if self.dr is not None and self.dr.ridge_prob > 0.0:
            if self.np_random.random() < self.dr.ridge_prob:
                # Get random torques between -max and +max
                ridge_torque_left  = self.np_random.uniform(
                    -self.dr.ridge_torque_max_nm,
                    self.dr.ridge_torque_max_nm
                )
                ridge_torque_right = self.np_random.uniform(
                    -self.dr.ridge_torque_max_nm,
                    self.dr.ridge_torque_max_nm
                )

                # Apply torques
                self.data.qfrc_applied[self._left_wheel_dof_idx]  = ridge_torque_left
                self.data.qfrc_applied[self._right_wheel_dof_idx] = ridge_torque_right

        # Optionally resample command mid-episode (if vel/yaw=0, save x,y location)
        if self.dr is not None and self.dr.cmd_resample_prob > 0.0:
            if self.np_random.random() < self.dr.cmd_resample_prob:
                if self.np_random.random() >= self.dr.cmd_zero_prob:
                    self._cmd_vel = float(self.np_random.uniform(*self.dr.cmd_vel_range))
                    self._cmd_yaw = float(self.np_random.uniform(*self.dr.cmd_yaw_range))
                else:
                    self._cmd_vel = 0.0
                    self._cmd_yaw = 0.0
                    self._cmd_zero_pos = np.array([self.data.qpos[0], self.data.qpos[1]])
                if self._debug:
                    print(f"New command: cmd_vel={self._cmd_vel:.3f}, cmd_yaw={self._cmd_yaw:.3f}")

        # Advance simulation by one step
        mujoco.mj_step(self.model, self.data)
        self._step += 1

        # Get observation
        obs = self._get_obs()
        pitch = obs[0]
        pitch_rate = obs[1]

        # Get actual forward velocity (privileged info) by projecting world-frame velocity onto the
        # chassis body frame X axis.
        xmat = self.data.xmat[self._chassis_id].reshape(3, 3)
        world_vel = self.data.cvel[self._chassis_id][3:6]
        vel_actual = float(np.dot(world_vel, xmat[:, 0]))

        # Get actual yaw rate (privileged info)
        yaw_actual = self.data.qvel[5]

        # Scale normalized commands to actual rates (rad/s)
        vel_target = self._cmd_vel * self.vel_max
        yaw_target = self._cmd_yaw * self.yaw_max

        # Penalties
        pitch_penalty = self.pitch_penalty_coef * pitch**2
        pitch_rate_penalty = self.pitch_rate_penalty_coef * pitch_rate**2
        action_penalty = self.action_penalty_coef * np.sum(action**2)
        action_smoothness_penalty = self.smoothness_penalty_coef * \
            np.sum((action - self._prev_action)**2)
        
        # Position penalty (from new "stand still" origin)
        # Only active when commanded to stand still
        if vel_target == 0.0 and yaw_target == 0.0:
            x_pos = self.data.qpos[0] - self._cmd_zero_pos[0]
            y_pos = self.data.qpos[1] - self._cmd_zero_pos[1]
            position_penalty = self.position_penalty_coef * (x_pos**2 + y_pos**2)
        else:
            position_penalty = 0.0

        # Save previous action
        self._prev_action = action.copy()

        # Gaussian reward: 1.0 for perfect tracking (no error) and decays smoothly toward 0 as error
        # grows. Sigma controls how quickly the reward decays with tracking error.
        vel_tracking_reward = self.vel_reward_coef * \
            math.exp(-((vel_actual - vel_target) ** 2) / self.sigma_vel)
        yaw_tracking_reward = self.yaw_reward_coef * \
            math.exp(-((yaw_actual - yaw_target) ** 2) / self.sigma_yaw)


        # Reward function:
        #   alive: encourage staying upright (not tipping over)
        #   pitch: penalty for leaning
        #   pitch_rate: penalty for rapid pitch changes (encourage smooth transitions)
        #   action: penalty for large motor commands (discourage aggressive controls)
        #   smoothness: penalty for changing mootor commands (discourage jitter)
        #   position: penalty for drifting from origin (only for stationary commands)
        #   vel_tracking: reward for matching the commanded forward velocity
        #   yaw_tracking: reward for matching the commanded yaw rate
        reward = self.alive_bonus \
            - pitch_penalty \
            - pitch_rate_penalty \
            - action_penalty \
            - action_smoothness_penalty \
            - position_penalty \
            + vel_tracking_reward \
            + yaw_tracking_reward

        # Termination (if robot tips or we run out of time in the episode)
        terminated = abs(pitch) > math.radians(self.tip_threshold_deg)
        truncated = self._step >= self.max_steps

        # DEBUG: print out command targets and rewards/penalties
        if self._debug and self._step % 1000 == 0 and self._step > 0:
            print(f"--- step {self._step} ---")
            print(f"  cmd_vel={self._cmd_vel:.3f}, vel_target={vel_target:.3f} m/s, vel_actual={vel_actual:.3f} m/s")
            print(f"  cmd_yaw={self._cmd_yaw:.3f}, yaw_target={yaw_target:.3f} rad/s, yaw_actual={yaw_actual:.3f} rad/s")
            print(f"  vel_tracking_reward={vel_tracking_reward:.4f} (perfect=1.0)")
            print(f"  yaw_tracking_reward={yaw_tracking_reward:.4f} (perfect=1.0)")
            print(f"  pitch={self._pitch:.3f} rad, pitch_penalty={pitch_penalty:.4f}")
            print(f"  pitch_rate={pitch_rate:.3f} rad/s, pitch_rate_penalty={pitch_rate_penalty:.4f}")
            print(f"  action_smoothness_penalty={action_smoothness_penalty:.4f}")

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