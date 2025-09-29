from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

from rotorpy.wind.default_winds import NoWind, ConstantWind
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_reward_functions import hover_reward 
# Other useful imports
import numpy as np                  # For array creation/manipulation
import matplotlib.pyplot as plt     # For plotting (optional)
from scipy.spatial.transform import Rotation  # For rotations (optional)
import os                           # For path generation
import gymnasium as gym
import pickle
import re                           # For parsing filenames (optional, for loading later)
from rotorpy.world import World
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
# Set a seed for reproducibility of random sampling (optional, uncomment for reproducibility)
# np.random.seed(42)
import torch
# --- Parameter Definitions for Data Generation ---

# Simulation parameters
sim_rate = 100 # Hz - Environment simulation rate
dt = 1/sim_rate # s - Timestep

# Data generation parameters
num_initial_conditions = 10          # Number of (v0, u0_base) combinations
num_wind_samples_per_initial_condition = 10 # Number of different wind conditions per (v0, u0_base)
steps_per_run = 1000                # Number of simulation steps per single wind condition run
total_data_points_aimed = num_initial_conditions * num_wind_samples_per_initial_condition * steps_per_run
print(f"Aiming for {total_data_points_aimed} data points.")

# Random walk parameters for attitude control
attitude_random_walk_std_dev = 0.02 # Radians, std dev of the Gaussian step for phi and theta random walk

# State and control dimensions for the learning problem
# s = [px, py, pz, vx, vy, vz] (6 states)
# v = [vx, vy, vz] (3 states) - Recorded at the start of the timestep
# w_wind = [wx, wy, wz] (3 states) - Wind vector in inertial frame, constant for a run
# u_applied = [u_theta_cmd, u_phi_cmd, u_T_scaled] (3 controls applied) - Note order: pitch, roll, thrust
# a = [ax, ay, az] (3 outputs) - Linear acceleration experienced by the drone

# Bounds for sampling initial state (velocity)
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

# Target Z position
target_z = 2 # meters

# Vehicle parameters (needed for thrust scaling)
vehicle = Multirotor(quad_params)
mass = vehicle.mass  # kg
min_total_thrust = vehicle.num_rotors * vehicle.k_eta * vehicle.rotor_speed_min**2
max_total_thrust = vehicle.num_rotors * vehicle.k_eta * vehicle.rotor_speed_max**2
min_upwards_acc = min_total_thrust / mass # Minimum possible upwards acceleration (can be negative)
max_upwards_acc = max_total_thrust / mass # Maximum possible upwards acceleration
a_g = 9.81 # gravity acceleration (positive downwards)

# --- Data Collection ---

# Initialize lists to store collected data
all_v = []          # Linear velocity [vx, vy, vz] at the start of the timestep
all_w_wind = []     # Wind velocity (inertial frame) [wx, wy, wz] for the run
all_u_applied = []  # Applied control [u_theta_cmd, u_phi_cmd, u_T_scaled]
all_a = []          # Resulting linear acceleration [ax, ay, az]
all_q = []          # Quaternion orientation [qx, qy, qz, qw] at the start of the timestep

print("Starting data generation with Z-PID control and random walk attitude...")

# Outer loop: Iterate over different initial velocities and base attitude commands
for i_cond_idx in range(num_initial_conditions):
    # 1. Sample initial velocity (v0) for this set of runs
    v0 = np.random.uniform(initial_velocity_bounds[:, 0], initial_velocity_bounds[:, 1])

    # 2. Sample initial base attitude commands (u_phi_base, u_theta_base) for this set of runs
    u_phi_base = np.random.uniform(attitude_control_bounds[0, 0], attitude_control_bounds[0, 1])
    u_theta_base = np.random.uniform(attitude_control_bounds[1, 0], attitude_control_bounds[1, 1])

    print(f"Initial Condition {i_cond_idx + 1}/{num_initial_conditions}: v0={np.round(v0,2)}, u_phi_base={u_phi_base:.2f}, u_theta_base={u_theta_base:.2f}")

    # Inner loop: Iterate over different wind conditions for the current (v0, u_phi_base, u_theta_base)
    for i_wind_idx in range(num_wind_samples_per_initial_condition):
        # print(f"  Starting Wind Sample {i_wind_idx + 1}/{num_wind_samples_per_initial_condition} for Initial Condition {i_cond_idx+1}") # Verbose

        # 3. Sample a constant wind vector for this specific run
        current_constant_wind_vector = np.random.uniform(wind_bounds[:, 0], wind_bounds[:, 1])
        wind_profile = ConstantWind(current_constant_wind_vector[0],
                                    current_constant_wind_vector[1],
                                    current_constant_wind_vector[2])

        # --- Instantiate and Reset Environment ---
        # Set initial position near target_z, default orientation, etc.
        p0 = np.array([0.0, 0.0, target_z + np.random.uniform(-0.1, 0.1)]) # Start slightly off target Z
        q0 = np.array([0.0, 0.0, 0.0, 1.0]) # quaternion [ix, iy, iz, r]
        w0_angular = np.array([0.0, 0.0, 0.0]) # angular velocity [wx, wy, wz]

        # Initial state dictionary for environment reset
        x0 = {'x': p0, 'v': v0, 'q': q0, 'w': w0_angular,
              'wind': np.array([0,0,0]), # This is a placeholder; wind_profile in make() is used.
              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])} # Hover speed

        # env = gym.make("Quadrotor-v0",
        #                num_envs=1,
        #                initial_states=x0,
        #                control_mode = 'cmd_ctatt', # Use scaled thrust and attitude commands
        #                reward_fn = hover_reward, # Not used for data logging itself
        #                quad_params = quad_params,
        #                max_time = steps_per_run * dt, # Set max time for this specific run (episode)
        #                wind_profile = wind_profile,
        #                world = None, # No obstacles
        #                sim_rate = sim_rate,
        #                render_mode="console") # No rendering for faster data generation
        env = QuadrotorEnv(
                num_envs=1,
                initial_states=x0,
                control_mode = 'cmd_ctatt', # Use scaled thrust and attitude commands
                reward_fn = hover_reward, # Not used for data logging itself
                quad_params = quad_params,
                max_time = steps_per_run * dt, # Set max time for this specific run (episode)
                wind_profile = wind_profile,
                world = None, # No obstacles
                sim_rate = sim_rate,
                render_mode="console") # No rendering for faster data generation


        observation = env.reset(options = {'initial_state': x0})

        current_time = 0.0
        e_integral_z = 0.0 # Reset Z-PID integral error for each run

        current_u_phi_rw = u_phi_base
        current_u_theta_rw = u_theta_base
        for step_idx in range(steps_per_run):
            state_current = {'x': observation[0][0:3],
                             'v': observation[0][3:6],
                             'q': observation[0][6:10],
                             'w_ang': observation[0][10:13]} # Angular velocity

            pos_current = state_current['x']
            v_current = state_current['v'] # This is the velocity at the start of the timestep
            q_current = state_current['q'] # This is the orientation at the start of the timestep
            current_z = pos_current[2]
            current_vz = v_current[2]

            # Get the wind velocity at the current state and time (for ConstantWind, this is fixed for the run)
            wind_at_vehicle_inertial = wind_profile.update(current_time, pos_current) # Should be current_constant_wind_vector

            # --- Z-PID Control Calculation (for u_T_scaled) ---
            e_pos_z = target_z - current_z
            e_vel_z = 0.0 - current_vz
            e_integral_z += e_pos_z * dt
            # Optional: Add integral windup protection here if needed

            acc_command_z = Kp_z * e_pos_z + Ki_z * e_integral_z + Kd_z * e_vel_z
            u_T_force_command = mass * (acc_command_z + a_g)

            if abs(max_total_thrust - min_total_thrust) < 1e-9:
                 u_T_scaled = 0.0
            else:
                 u_T_scaled = np.interp(u_T_force_command,
                                        [min_total_thrust, max_total_thrust],
                                        [-1.0, 1.0])
            u_T_scaled = np.clip(u_T_scaled, -1.0, 1.0)

            # --- Random Walk for Attitude Control Inputs (u_phi, u_theta) ---
            phi_increment = np.random.normal(0, attitude_random_walk_std_dev)
            theta_increment = np.random.normal(0, attitude_random_walk_std_dev)

            current_u_phi_rw += phi_increment
            current_u_theta_rw += theta_increment

            # Boundary Check and Re-sampling if out of bounds
            if not (attitude_control_bounds[0, 0] <= current_u_phi_rw <= attitude_control_bounds[0, 1]):
                current_u_phi_rw = np.random.uniform(attitude_control_bounds[0, 0], attitude_control_bounds[0, 1])

            if not (attitude_control_bounds[1, 0] <= current_u_theta_rw <= attitude_control_bounds[1, 1]):
                current_u_theta_rw = np.random.uniform(attitude_control_bounds[1, 0], attitude_control_bounds[1, 1])

            # Final commanded attitude for this step
            u_phi_cmd = current_u_phi_rw
            u_theta_cmd = current_u_theta_rw
            u_psi_cmd = 0.0 # Fixed yaw command

            # Construct the action for the simulator [u_T_scaled, u_phi_cmd, u_theta_cmd, u_psi_cmd]
            # Note: rotorpy's cmd_ctatt expects (thrust_scale, roll_cmd, pitch_cmd, yaw_cmd/yaw_rate_cmd)
            action = np.array([u_T_scaled, u_phi_cmd, u_theta_cmd, u_psi_cmd])

            # --- Step Simulation ---
            observation_next, dynamics, terminated, truncated, info = env.step(action)

            # Extract linear acceleration from dynamics (dynamics[:3] is linear acceleration [ax, ay, az])
            a_current_inertial = dynamics[:3] # This is the acceleration resulting from action at state_current

            # --- Store Data Point ---
            all_v.append(v_current) # Velocity at the start of the timestep
            all_w_wind.append(wind_at_vehicle_inertial) # Wind vector for the run (inertial frame)
            all_u_applied.append(np.array([u_theta_cmd, u_phi_cmd, u_T_scaled])) # Applied controls (pitch, roll, thrust_scaled)
            all_a.append(a_current_inertial) # Resulting acceleration
            all_q.append(q_current) # Quaternion at the start of the timestep

            # Update for the next step
            observation = observation_next
            current_time += dt

            if terminated or truncated:
                if terminated:
                    print(f"    Run terminated at step {step_idx+1}.")
                    print(f"    Termination Info: {info}") # <-- PRINT INFO HERE FOR TERMINATION
                if truncated:
                    print(f"    Run truncated (max_time reached) at step {step_idx+1}.")
                    print(f"    Truncation Info: {info}") # <-- PRINT INFO HERE FOR TRUNCATION
                # You could also print it regardless of termination/truncation for debugging every step:
                # print(f"Step {step_idx+1} Info: {info}")
                break # Exit the step loop

        # Close the environment after each run (corresponding to one wind condition)
        env.close()
    # End of wind_samples loop
    if (i_cond_idx + 1) % 1 == 0: # Print progress every initial condition
        print(f"  Finished all wind samples for Initial Condition {i_cond_idx + 1}. Total data points so far: {len(all_v)}")

# End of initial_conditions loop
print("Data generation complete.")

# --- Convert Lists to NumPy Arrays ---
all_v = np.array(all_v)
all_w = np.array(all_w_wind)
all_u = np.array(all_u_applied)
all_a = np.array(all_a)
all_q = np.array(all_q)

print(f"Collected {len(all_v)} data points.")
print(f"Shape of v data (velocities): {all_v.shape}")
print(f"Shape of w_wind data (wind vectors): {all_w.shape}")
print(f"Shape of u_applied data (controls): {all_u.shape}") # Stored as [theta, phi, T_scaled]
print(f"Shape of a data (accelerations): {all_a.shape}")
print(f"Shape of q data (quaternions): {all_q.shape}")


# --- Save Data ---
data_dictionary = {
    'velocities': all_v,          # [vx, vy, vz] (N x 3)
    'winds': all_w,               # [wx, wy, wz] (N x 3)
    'controls': all_u,            # [u_T_scaled, u_phi, u_theta] (N x 3)
    'accelerations': all_a,       # [ax, ay, az] (N x 3)
    'quaternions': all_q,         # [qx, qy, qz, qw] (N x 4)
    'metadata': {
        'description': 'Dataset for quadrotor dynamics. For each (initial_velocity, initial_base_attitude), data is collected under multiple wind conditions. Attitude commands (roll, pitch) undergo a random walk from the base, with re-sampling if bounds are exceeded. Z-altitude is PID controlled.',
        'num_initial_conditions': num_initial_conditions,
        'num_wind_samples_per_initial_condition': num_wind_samples_per_initial_condition,
        'steps_per_run': steps_per_run, # Steps per single wind condition
        'total_data_points_collected': len(all_v),
        'total_data_points_aimed': total_data_points_aimed,
        'attitude_random_walk_std_dev': attitude_random_walk_std_dev,
        'initial_velocity_bounds': initial_velocity_bounds.tolist(), # Convert numpy arrays for JSON compatibility if needed, pickle handles them fine
        'wind_bounds': wind_bounds.tolist(),
        'attitude_control_bounds': attitude_control_bounds.tolist(), # Bounds for [u_phi, u_theta]
        'z_pid_gains': {'Kp_z': Kp_z, 'Ki_z': Ki_z, 'Kd_z': Kd_z},
        'target_z': target_z,
        'sim_timestep': dt,
        'vehicle_params': quad_params, # This is a dictionary, fine for pickle
        'control_storage_order': '[commanded_pitch (u_theta), commanded_roll (u_phi), scaled_total_thrust (u_T_scaled)]'
    }
}

# Define filename and path
filename = f'quadrotor_dynamics_varied_wind_rw_control_ic{num_initial_conditions}_ws{num_wind_samples_per_initial_condition}_near_zero_wind.pkl'
filepath = os.path.join(os.getcwd(), filename)

# Save the dictionary using pickle
try:
    with open(filepath, 'wb') as f:
        pickle.dump(data_dictionary, f)
    print(f"Successfully saved data to '{filename}'.")
except Exception as e:
    print(f"Error saving data to '{filename}': {e}")

# --- Example of Loading Data (Optional) ---
print("\n--- Example of Loading Data ---")
try:
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)

    print(f"Successfully loaded data from '{filename}'.")
    print("Keys available:", loaded_data.keys())
    # print("Metadata:", loaded_data['metadata']) # Can be verbose
    print("Description from metadata:", loaded_data['metadata']['description'])
    print("Shape of loaded velocities:", loaded_data['velocities'].shape)
    print("Shape of loaded accelerations:", loaded_data['accelerations_inertial'].shape)
    print("Number of data points collected:", loaded_data['metadata']['total_data_points_collected'])


except FileNotFoundError:
    print(f"Error loading data: File '{filename}' not found.")
except Exception as e:
    print(f"Error loading data: {e}")