import os, sys

from matplotlib.colors import Normalize, from_levels_and_colors
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
log_dir = interm_dir + '010122140144'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
# env_config = {}
env_config = {"motor_control_mode":"CARTESIAN_PD",
               "task_env": "LR_COURSE_TASK",
               "observation_space_mode": "LR_COURSE_OBS"}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
env_config['competition_env'] = False

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

num_steps = 2000

States = np.zeros((num_steps, 84))
Forces_n = np.zeros((num_steps, 4))

Vel = np.zeros((num_steps, 1))
Foot_z = np.zeros((num_steps, 4))
Energy = np.zeros((num_steps, 1))
Contact_bool = np.zeros((num_steps, 4))

q_prev = np.zeros(12)
foot_p = np.zeros(12)
idx_init = 0

mass = 12.454
g_acc = 9.8

vel_avg = []
foot_z_avg = []
cot_avg = []
x_base_avg = []
duty_factor_avg = []

for i in range(num_steps):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards

    Vel[i, 0] = np.linalg.norm(env.envs[0].env.robot.GetBaseLinearVelocity())

    for j in range(4):
        _, foot_p[3*j:3*j+3] = env.envs[0].env.robot.ComputeJacobianAndPosition(j)

    Foot_z[i, :] = foot_p[2:12:3]
    robot_base = np.array(env.envs[0].env.robot.GetBasePosition())
    robot_height = robot_base[2]
    Foot_z[i, :] += robot_height

    q = np.array(env.envs[0].env.robot.GetMotorAngles())
    q_dot = q - q_prev
    q_prev = q
    torques = env.envs[0].env.robot.GetMotorTorques()
    Energy[i, 0] = np.dot(q_dot, torques)

    Contact_bool[i, :] = np.array(env.envs[0].env.robot.GetContactInfo()[3])

    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

        final_base = np.array(info[0]['base_pos'])

        vel = Vel[idx_init:i]
        foot_z = Foot_z[idx_init:i]
        energy = Energy[idx_init+1:i]
        contact_bool = Contact_bool[idx_init:i]

        vel_avg.append(np.mean(vel))
        foot_z_avg.append(np.mean(foot_z, axis=0))

        x_base_avg.append(final_base[0])
        
        dist = np.linalg.norm(final_base[0:2])
        cot_avg.append(np.sum(energy)/mass/g_acc/dist)

        duty_factor_avg.append(np.mean(contact_bool, axis=0))

        idx_init = i

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    States[i, 0:3] = env.envs[0].env.robot.GetBasePosition()
    States[i, 3:84] = env.envs[0].env._observation

    # Forces_n[i, :] = env.envs[0].env.robot.GetContactInfo()[2]

    # if i == 500:
    #   # env.envs[0].env._cmd_base_vel_normed = np.array([np.cos(np.pi/6*2), np.sin(np.pi/6*2), 0.0])
    #   env.envs[0].env._cmd_base_vel_normed = np.array([np.cos(np.pi/6*3), np.sin(np.pi/6*3), 0.0])
    #   print(f'>>>>>>>>>> Current cmd vel is: {np.array2string(env.envs[0].env._cmd_base_vel_normed, precision=4, separator=",")}')
    # elif i == 1000:
    #   env.envs[0].env._cmd_base_vel_normed = np.array([np.cos(np.pi/6*0), np.sin(np.pi/6*0), 0.0])
    #   print(f'>>>>>>>>>> Current cmd vel is: {np.array2string(env.envs[0].env._cmd_base_vel_normed, precision=4, separator=",")}')
    # elif i == 1500:
    #   env.envs[0].env._cmd_base_vel_normed = np.array([np.cos(np.pi/6*3), np.sin(np.pi/6*3), 0.0])
    #   print(f'>>>>>>>>>> Current cmd vel is: {np.array2string(env.envs[0].env._cmd_base_vel_normed, precision=4, separator=",")}')
    
# [TODO] make plots:
time_step = env.envs[0].env._time_step
t = np.arange(num_steps)*time_step

Base_pos = States[:, 0:3]
base_x = States[:, 0]
base_y = States[:, 1]

fig, axs = plt.subplots(1, 2)
axs[0].plot(t, base_x, label='X')
axs[0].plot(t, base_y, label='Y')
axs[0].set_title('Base Position X/Y-Time')
axs[0].legend()

axs[1].plot(base_x, base_y)
axs[1].set_title('Base Position Y-X')

plt.show()

ofile = open('robot_states_rl.csv', 'wb')
np.savetxt(ofile, States, delimiter=',')
ofile.close()

print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(vel_avg)
print(np.mean(vel_avg))
print(foot_z_avg)
print(np.mean(foot_z_avg, axis=0))
print(x_base_avg)
print(np.mean(x_base_avg))
print(cot_avg)
print(np.mean(cot_avg))
print(duty_factor_avg)
print(np.mean(duty_factor_avg, axis=0))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

# print('\n\n')
# print('Max Normal Force: ')
# print(np.amax(Forces_n, axis=0))
# print(np.mean(Forces_n, axis=0))
# print('\n\n')

# ofile = open('normal_forces.csv', 'wb')
# np.savetxt(ofile, Forces_n, delimiter=',')
# ofile.close()

# max normal forces
# [911 842 952 776]
# [1.19e+03 748 1.25e+03 633]
# [1.03e+03 822 723 938]
# average...
# [37 28.4 32 37.3]
