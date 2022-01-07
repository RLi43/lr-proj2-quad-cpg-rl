'''
Author: Chengkun Li
LastEditors: Chengkun Li
Date: 2021-12-01 02:23:02
LastEditTime: 2022-01-07 14:48:40
Description: Modify here please
FilePath: /lr-proj2-quad-cpg-rl/load_sb3.py
'''
import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
from loguru import logger

from torch._C import dtype
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

# logger
import logging
logging.getLogger().setLevel(logging.INFO)

# stable baselines
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.cmd_util import make_vec_env

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = ""
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '111121133812'
log_dir = interm_dir + ''

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {"motor_control_mode":"CARTESIAN_PD",
               "task_env": "LR_COURSE_TASK",
               "observation_space_mode": "LR_COURSE_OBS"}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
env_config['test_env'] = False
env_config['competition_env'] = False
env_config['dy_rand'] = False # for training! only for validation!



plot_monitor = True


# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
if plot_monitor:
  monitor_results = load_results(log_dir)
  logging.info(monitor_results)
  plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
  plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0



leg_name = ['FR', 'FL', 'RR', 'RL']

steps = 5000
# Plot only one trial
only_once = True
base_linear = np.zeros([steps, 3])
base_angular = np.zeros([steps, 3])
motor_angles = np.zeros([steps, 4, 3])
foot_pos = np.zeros([steps, 4, 3])
contact_info = np.zeros([steps, 4])
base_pos = np.zeros([steps, 3])
motor_angles = np.zeros([steps, 4, 3])
motor_torques = np.zeros([steps, 4, 3])

# for calculation of COT
q_hist = [0]*12
energy = 0
# start index for a test trail
start_i = 0
distance = 0
x, y = 0, 0
x_prev, y_prev = 0, 0
dist_per_trail = []
hist_COT = []
step_energy_curve = []

for i in range(steps):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    # logging.info(type(_states))
    # logger.info(action)
    obs, rewards, dones, info = env.step(action)
    
    base_pos[i, :] = env.envs[0].env.robot.GetBasePosition()
    """
    obs: 1x47
    self._observation = np.concatenate((
        self.robot.GetBaseLinearVelocity(), # 3x1
        self.robot.GetBaseAngularVelocity(), # 3x1
        self.robot.GetMotorAngles(), # 12x1
        self.robot.GetMotorVelocities(), # 12x1
        self.robot.GetMotorTorques(), # 12x1
        self.robot.GetBaseOrientation(), # 4x1
        foot_pos, # 12x1
        foot_vel, # 12x1
        np.array(self.robot.GetContactInfo()[3]) # 4x1
      ))
    """
    # logger.info('Current speed: {}, normed: {}'.format(tmp[0:3], np.linalg.norm(tmp[0:3])))
    base_linear[i, :] = env.envs[0].env.robot.GetBaseLinearVelocity()
    # logger.info('speed vector: {}'.format(base_linear[i, :]))
    base_angular[i, :] = env.envs[0].env.robot.GetBaseAngularVelocity()
    motor_angles[i, :, :] = env.envs[0].env.robot.GetMotorAngles().reshape(4, 3)
    motor_torques[i, :, :] = env.envs[0].env.robot.GetMotorTorques().reshape(4, 3)
    
    foot_pos_tmp = []
    for legid in range(4):
      _, pos = env.envs[0].env.robot.ComputeJacobianAndPosition(legid)
      foot_pos_tmp += pos.tolist()
    foot_pos[i, :, :] = np.array(foot_pos_tmp).reshape(4, 3)    
    contact_info[i, :] = np.array(env.envs[0].env.robot.GetContactInfo()[3])
    episode_reward += rewards

    # Calculate energy
    q = motor_angles[i, :, :].ravel()
    step_energy = sum((q - q_hist) * motor_torques[i, :, :].ravel())
    # logger.debug("{}, {}, {}".format((q - q_hist),  motor_torques[i, :, :].ravel(), (q - q_hist) * motor_torques[i, :, :].ravel()))
    
    # ISSUES: there are abnormal energy values at the beginning of control
    # current solution: remove them from the total energy calculation
    step_energy_curve.append(step_energy)
    if step_energy < -20:
      logger.debug('step energy less than -20!: {}'.format(step_energy))
    if i == start_i:
      logger.debug('step_energy at i={}: {}'.format(start_i, step_energy))
      step_energy = 0
    energy += step_energy
    
    # if np.array_equal(q_hist, [0]*12):
    #   logger.debug('q_hist is reset to 0')
    # logger.info('Current dq: {}; current torque: {}; product: {}'.format(q - q_hist, motor_torques[i, :, :].ravel(), (q - q_hist) * motor_torques[i, :, :].ravel()))
    # logger.info('Current consumed energy: {}'.format(energy))
    q_hist = q
    x, y = base_pos[i, 0], base_pos[i, 1]
    if i > 0:
      dx, dy = x - x_prev, y - y_prev  
    else:
      dx, dy = 0, 0
    x_prev, y_prev = x, y
    distance += np.sqrt(dx**2 + dy**2)
    # logger.info('robot contact normal force: {}'.format(env.envs[0].env.robot.GetContactInfo()[2]))
    if dones:
        logger.info('episode_reward: {}'.format(episode_reward))
        logger.info('Total energy is {}'.format(energy))
        logger.info('Total distance traveled: {}', distance)
        COT = energy/distance
        hist_COT.append(COT)
        logger.info('Current mean of COT = {}; COT of this trail: {}'.format(np.mean(hist_COT), COT))
        logger.info('Final base position: {}'.format(info[0]['base_pos']))
        dist_per_trail.append(info[0]['base_pos'][0])
        logger.info('Current mean of end position: {}'.format(np.mean(dist_per_trail)))
        for i in range(4):
          avg_height = np.mean(base_pos[:steps, 2] + foot_pos[:steps, i, 2])
          logger.info('Average foot height of {}: {}'.format(leg_name[i], avg_height))
        episode_reward = 0
        energy = 0
        distance = 0
        start_i = i+1
        q_hist = [0]*12
        
        
        if only_once:
          steps = i
          break
    






"""
Foot order: FR, FL, RR, RL
"""
# Plot speed
fig, ax = plt.subplots(nrows=4, constrained_layout=True, sharex=True)
t = np.arange(steps)
ax[0].plot(t, base_linear[:steps, 0], label='X speed')
ax[0].legend()
ax[1].plot(t, base_linear[:steps, 1], label='Y speed')
ax[1].legend()
ax[2].plot(t, base_linear[:steps, 2], label='Z speed')
ax[2].legend()
ax[3].plot(t, np.sqrt(base_linear[:steps, 0]**2 + base_linear[:steps, 1]**2 + base_linear[:steps, 2]**2)\
, label='Total speed')
avg = np.mean(np.sqrt(base_linear[:steps, 0]**2 + base_linear[:steps, 1]**2))
ax[3].plot(t, np.ones([steps]) * avg, label='Average speed')
ax[3].set(title='Average speed is {}'.format(avg))
ax[3].legend()
ax[3].set_xlabel('Time steps')

# Plot foot contact information
fig, ax = plt.subplots(nrows=2, sharex=True)

ax[1].set_xlabel('Time steps')
for i in range(2):
  ax[0].plot(t, contact_info[:steps, i], label='Contact information of {}'.format(leg_name[i]))
ax[0].legend()

for i in range(2, 4):
  ax[1].plot(t, contact_info[:steps, i], label='Contact information of {}'.format(leg_name[i]))
ax[1].legend()


# Plot base position
if only_once:
  fig, ax = plt.subplots()
  ax.plot(base_pos[:steps, 0], base_pos[:steps, 1])
  ax.set(title='Base position of Legged robot')
  ax.scatter(base_pos[0, 0], base_pos[0, 1], s=120, c='r', marker='X', label='Start Point')
  logger.info((base_pos[0, 0], base_pos[0, 1]))
  ax.scatter(base_pos[steps-1, 0], base_pos[steps-1, 1], s=120, c='g', marker='*', label='End Point')
  ax.legend()
  logger.info((base_pos[steps-1, 0], base_pos[steps-1, 1]))

# Plot foot position
fig, ax = plt.subplots()
for i in range(4):
  ax.scatter(base_pos[:steps, 0] + foot_pos[:steps, i, 0], \
    base_pos[:steps, 1] + foot_pos[:steps, i, 1],\
       label='Foot trajectory of {}'.format(leg_name[i]), s=3)
ax.set(title='Foot trajectories of all trials', xlabel='x', ylabel='y')
ax.set_ylim([-30, 30])
ax.legend()

fig, ax = plt.subplots()
ax.bar(np.arange(1, len(dist_per_trail)+1), dist_per_trail)
ax.set(title='Distance per trail', xlabel='Trail number')

# Calculate duty cycyle
for i in range(4):
  logger.info('Duty cycle of {} is {}'.format(leg_name[i], np.count_nonzero(contact_info[:steps, i])/(steps)))

# Plot foot height
fig, ax = plt.subplots(nrows=4, sharex=True)
for i in range(4):
  ax[i].plot(t, base_pos[:steps, 2] + foot_pos[:steps, i, 2], label='Foot height of {}'.format(leg_name[i]))
  avg_height = np.mean(base_pos[:steps, 2] + foot_pos[:steps, i, 2])
  ax[i].plot(t, np.ones([steps]) * avg_height, label='Average height is {}'.format(avg_height))
  ax[i].set(title='Foot heights of {}'.format(leg_name[i]), xlabel='steps', ylabel='height')
# ax.set_ylim([-30, 30])
  ax[i].legend()

# Plot energy curve
fig, ax = plt.subplots()
ax.plot(t, step_energy_curve[1:], label='Step Energy')
ax.set(title='Energy curve', xlabel='steps', ylabel='Energy')
plt.tight_layout()
plt.show()
