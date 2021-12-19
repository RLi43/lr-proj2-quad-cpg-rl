import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

# stable baselines
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.cmd_util import make_vec_env

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "PPO"
# LEARNING_ALG = "SAC"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '111121133812'
log_dir = interm_dir + '121921103954'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
# env_config = {}
env_config = {"motor_control_mode":"CARTESIAN_PD",
               "task_env": "LR_COURSE_TASK",
               "observation_space_mode": "LR_COURSE_OBS"}
env_config['render'] = True
env_config['record_video'] = True
env_config['add_noise'] = False 
env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
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

# [TODO] initialize arrays to save data from simulation 
# Base_pos = np.zeros((2000, 3))
# Base_vel = np.zeros((2000, 3))
# Motor_ang = np.zeros((2000, 12))
# Motor_vel = np.zeros((2000, 12))
# Motor_torq = np.zeros((2000, 12))
# Base_orient = np.zeros((2000, 4))
States = np.zeros((2000, 73))

for i in range(2000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    States[i, 0:3] = env.envs[0].env.robot.GetBasePosition()
    States[i, 3:73] = env.envs[0].env._observation
    
# [TODO] make plots:
time_step = env.envs[0].env._time_step
t = np.arange(2000)*time_step

Base_pos = States[:, 0:3]
base_x = States[:, 0]
base_y = States[:, 1]

fig, axs = plt.subplots(1, 2)
axs[0].plot(t, base_x, label='X')
axs[0].plot(t, base_y, label='Y')
axs[0].set_title('Base Position X/Y-Time')

axs[1].plot(base_x, base_y)
axs[1].set_title('Base Position Y-X')
