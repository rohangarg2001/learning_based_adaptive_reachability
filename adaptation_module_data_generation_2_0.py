## new file as the older file was for rotorpy 1.0 and now they have updated to rotorpy 2.0
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from rotorpy.controllers.quadrotor_control import BatchedSE3Control
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.trajectories.hover_traj import BatchedHoverTraj
from rotorpy.vehicles.multirotor import BatchedMultirotorParams

from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_environments import make_default_vec_env  # Helper script

num_quads = 1
sim_rate = 100  # Hz - Environment simulation rate
dt = 1/sim_rate

num_initial_conditions = 10          # Number of (v0, u0_base) combinations
num_wind_samples_per_initial_condition = 10 # Number of different wind conditions per (v0, u0_base)
steps_per_run = 1000                # Number of simulation steps per single wind condition run
total_data_points_aimed = num_initial_conditions * num_wind_samples_per_initial_condition * steps_per_run
print(f"Aiming for {total_data_points_aimed} data points.")

# Random walk parameters for attitude control
attitude_random_walk_std_dev = 0.02 # Radians, std dev of the Gaussian step for phi and theta random walk
# State and control dimensions for the learning problem
# s = [px, py, pz, vx, vy, vz] (6
# v = [vx, vy, vz] (3 states) - Recorded at the start of the timestep
# w_wind = [wx, wy, wz] (3 states) - Wind vector in inertial frame, constant for a run
# u_applied = [u_theta_cmd, u_phi_cmd, u_T_scaled] (3 controls applied) - Note order: pitch, roll, thrust
# a = [ax, ay, az] (3 outputs) - Linear acceleration experienced by the drone

initial_velocity_bounds = np.array([
    [-2.0, 2.0],  # vx bounds [min, max] in m/s
    [-2.0, 2.0],  # vy bounds [min, max] in m/s
    [-0.1, 0.1]   # vz bounds [min, max] in m/s - Keep vertical initial velocity small for z-PID stability
])

# Bounds for sampling constant wind vector for each run
wind_bounds = np.array([
    [-0.2, 0.2],  # wx bounds [min, max] in m/s
    [-0.2, 0.2],  # wy bounds [min, max] in m/s
    [-0.02, 0.02]   # wz bounds [min, max] in m/s (typically less vertical wind)
])

# Bounds for sampling attitude control inputs u_model = [u_phi, u_theta]
# These bounds are also used for re-sampling if the random walk goes out of bounds.
attitude_control_bounds = np.array([
    [-0.25, 0.25],    # u_phi bounds [min, max] radians (approx +/- 14.3 degrees)
    [-0.25, 0.25]     # u_theta bounds [min, max] radians (approx +/- 14.3 degrees)
])
# u_psi (commanded yaw angle) will be set to 0
# u_T_scaled will be determined by the Z-PID controller

# Z-PID Controller Gains (Tune these for stable altitude hold)
Kp_z = 1.0  # Proportional gain for z-position error
Ki_z = 0.1  # Integral gain for z-position error
Kd_z = 1.5  # Derivative gain for z-velocity error

# target in z 
target_z = 2            # meters

vehicle