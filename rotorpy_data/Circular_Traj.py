# The simulator is instantiated using the Environment class
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control

# And a trajectory generator
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj, ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.minsnap import MinSnap

# You can optionally specify a wind generator, although if no wind is specified it will default to NoWind().
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind
from rotorpy.wind.dryden_winds import DrydenGust, DrydenGustLP
from rotorpy.wind.spatial_winds import WindTunnel

# You can also optionally customize the IMU and motion capture sensor models. If not specified, the default parameters will be used. 
from rotorpy.sensors.imu import Imu
from rotorpy.sensors.external_mocap import MotionCapture

# You can also specify a state estimator. This is optional. If no state estimator is supplied it will default to null. 
try:
      from rotorpy.estimators.wind_ukf import WindUKF
except:
      print("FilterPy is not installed in the basic version of rotorpy. Please install the filter version of rotorpy by running pip install rotorpy[filter]")

# Also, worlds are how we construct obstacles. The following class contains methods related to constructing these maps. 
from rotorpy.world import World

# Import the QuadrotorEnv gymnasium
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_reward_functions import hover_reward

# Other useful imports
import numpy as np                  # For array creation/manipulation
import matplotlib.pyplot as plt     # For plotting, although the simulator has a built in plotter
from scipy.spatial.transform import Rotation  # For doing conversions between different rotation descriptions, applying rotations, etc. 
import os                           # For path generation
import gymnasium as gym
import pickle
np.random.seed(42)

# Define wind basics
wind_min = 0
wind_max = 10

#def wind_parameters(wind_min, wind_max):
#  while True:
#    wx, wy, wz = np.random.uniform(-wind_max, wind_max, 3)
#    magnitude = np.sqrt(wx**2 + wy**2 + wz**2)
#    if wind_min <= magnitude and magnitude <= wind_max:
#      return [wx, wy, wz]

# sample the overall vector [wx, wy, wz]
def wind_parameters(max_wind):
  '''
  sample a wind vector uniformly from a sphere of radius max_wind
  '''
  # sample a random point on a unit sphere
  phi = np.random.uniform(0, 2*np.pi)
  costheta = np.random.uniform(-1, 1)
  theta = np.arccos(costheta)
  x = np.sin(theta) * np.cos(phi)
  y = np.sin(theta) * np.sin(phi)
  z = np.cos(theta)
  unit_vector = np.array([x, y, z])
  # sample a random radius r in [0, max_wind]
  r = np.random.uniform(0, max_wind)
  vector = r * unit_vector
  wx, wy, wz = vector
  return [wx, wy, wz]

# define trajectory
trajectory = CircularTraj(center = np.array([0,0,0]),
                          radius = 3,                 # can vary [1, 2, 3]
                          freq = 0.2,                 # can vary [0.1, 0.2, 0.25]
                          yaw_bool = False,
                          direction = 'CCW')
# initial state
x0 = {'x': trajectory.update(0)['x'],  # starts at x = center_x + radius, y = center_y
      'v': trajectory.update(0)['x_dot'],
      'q': np.array([0, 0, 0, 1]),
      'w': np.zeros(3,),
      'wind': np.array([0, 0, 0]),
      'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

## define trajectory basics
#trajectory_type = 'Circular'
#trajectory_parameter = {'center': np.array([0,0,0]),
#                        'radius': np.array([1,2,3]),  
#                        'frequency': np.array([0.1,0.2,0.25])}
## define trajectory
#i = 0
#if trajectory_type == 'Circular':
#  center = trajectory_parameter['center']
#  radius = trajectory_parameter['radius'][i]
#  freq = trajectory_parameter['frequency'][i]
#  trajectory = CircularTraj(center = center, radius = radius, freq = freq, yaw_bool = False, direction = 'CCW')
#  # initial state
#  x0 = {'x': trajectory.update(0)['x'],  # starts at x = center_x + radius, y = center_y
#        'v': trajectory.update(0)['x_dot'],
#        'q': np.array([0, 0, 0, 1]),
#        'w': np.zeros(3,),
#        'wind': np.array([0, 0, 0]),
#        'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

# define vehicle and controller
vehicle = Multirotor(quad_params,
                     initial_state = x0,
                     control_abstraction = 'cmd_ctatt')
controller = SE3Control(quad_params)

# define simulation time
T = 5  # s
dt = 0.01  # s
times = np.arange(0, T+dt, dt)  # [0, dt, 2dt,..., T]

# initialize a dictionary to store data
data = {'wind': [],
        's': [],
        's_dot': [],
        'u': []}
num_states = 6    # px, py, pz, vx, vy, vz
num_controls = 3  # u_T, u_phi, u_theta

# define gravitational constants
a_g = 9.81
K_p = 0.5  # need to decide
K_d = 0.5  # need to decide
K_i = 0    # need to decide
e_integral = 0

num_conditions = 100

for k in range(num_conditions):  # num_conditions different wind condition

  print("Wind condition: ", k)
  
  # get wind parameters
  wx, wy, wz = wind_parameters(wind_max)
  print("Start simulating...wx = ", wx, ", wy = ", wy, ", wz = ", wz)
  # define wind
  wind = ConstantWind(wx, wy, wz)
  data['wind'].append(np.array([wx, wy, wz]))

  # set gymnasium environment
  env = gym.make("Quadrotor-v0",
                 control_mode = 'cmd_ctatt',
                 reward_fn = hover_reward,
                 quad_params = quad_params,
                 max_time = 10,
                 wind_profile = wind,
                 world = None,
                 sim_rate = 100,
                 render_mode= None,
                 render_fps=30)
  observation, info = env.reset(options = {'initial_state': x0})

  # initialize s, s_dot, u arrays to store trajectory data
  s = np.zeros((len(times), num_states))      # px, py, pz, vx, vy, vz  
  s_dot = np.zeros((len(times), num_states))  # vx, vy, vz, ax, ay, az
  u = np.zeros((len(times), num_controls))    # u_T, u_phi, u_theta


  # store initial states
  s[0, :] = np.concatenate((x0['x'], x0['v']))

  for i, t in enumerate(times[:-1]): # iterate from 0s to (T-dt)s
    # Unpack the observation from the environment
    state = {'x': observation[0:3],
             'v': observation[3:6],
             'q': observation[6:10],
             'w': observation[10:13]}
    current_pos = state['x']
    current_vel = state['v']

    # extract desired acceleration for the current state 
    desired_state = trajectory.update(t)
    desired_pos = desired_state['x']
    desired_vel = desired_state['x_dot']
    desired_acc = desired_state['x_ddot']
    
    # determine the control commands for the reduced-order system
    e_pos = desired_pos - current_pos
    e_vel = desired_vel - current_vel
    e_integral += e_pos * dt
    acc = desired_acc + K_p * e_pos + K_d * e_vel + K_i * e_integral
    #a_x, a_y, a_z = desired_acc[:]
    a_x, a_y, a_z = acc[:]
    u_theta = np.arctan(a_x/a_g) # a_x = a_g*tan(u_theta)
    u_phi = np.arctan(-a_y/a_g)  # a_y = -a_g*tan(u_phi)
    u_T = a_z + a_g              # a_z = u_T - a_g  # u_T is in m/s^2!!
    # convert u_T from m/s^2 to N
    mass = vehicle.mass  # kg
    u_T_force = mass * u_T     # N

    # rescale net thrust u_T
    max_thrust = vehicle.k_eta * vehicle.rotor_speed_max**2
    min_thrust = vehicle.k_eta * vehicle.rotor_speed_min**2
    num_rotors = vehicle.num_rotors
    u_T_scaledforce = np.interp(u_T_force, [num_rotors*min_thrust, num_rotors*max_thrust], [-1, 1])

    # construct action
    action = np.array([u_T_scaledforce, u_phi, u_theta, 0])

    # step simulation
    observation, dynamics, _, _, _ = env.step(action)

    # store s, s_dot, and u information
    s[i+1, :] = observation[0:6]  # next state pos and vel after applying the commands
    s_dot[i, :] = np.concatenate((current_vel, dynamics[:3]))  # current state dynamics after applying control command 
                                                                 # (current state vel and linear acc vdot)    
    u[i, :] = np.array([u_theta, u_phi, u_T])  # applied command
        
  # end simulation
  env.close()

  # store data to the dictionary
  data['s'].append(s)
  data['s_dot'].append(s_dot)
  data['u'].append(u)

  print("Finish simulating...")

#  (fig, ax) = plt.subplots(nrows=1, ncols=1, num="Trajectory in XY Plane")
#  ax.plot(s[:, 0], s[:, 1])
#  ax.set_xlabel("X Position, m")
#  ax.set_ylabel("Y Position, m")

#  (fig, ax) = plt.subplots(nrows=1, ncols=1, num="Trajectory in XZ Plane")
#  ax.plot(s[:, 0], s[:, 2])
#  ax.set_xlabel("X Position, m")
#  ax.set_ylabel("Z Position, m")

#  (fig, ax) = plt.subplots(nrows=1, ncols=1, num="Trajectory in YZ Plane")
#  ax.plot(s[:, 1], s[:, 2])
#  ax.set_xlabel("Y Position, m")
#  ax.set_ylabel("Z Position, m")

#  plt.show()


# save the dictionary to a .pkl file
filename = 'R30F020.pkl'
if os.path.exists(filename):
      user_input = input(f"File '{filename}' already exists. Overwrite? (yes/no): ").strip().lower()
      if user_input != 'yes':
            raise FileExistsError(f"File '{filename}' already exists. Operation aborted.")

with open(filename, 'wb') as f:
  pickle.dump(data, f)
  
print(f"Dictionary saved as '{filename}'.")
