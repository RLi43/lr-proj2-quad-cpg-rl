'''
Author: Chengkun Li
LastEditors: Chengkun Li
Date: 2021-11-30 01:47:27
LastEditTime: 2021-12-28 20:28:46
Description: Legged Robot Project 2 CPG & HOPF Network part
FilePath: /lr-proj2-quad-cpg-rl/hopf_network.py
'''
"""
CPG in polar coordinates based on: 
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import time
import argparse
from loguru import logger
import numpy as np
import matplotlib
from sys import platform
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.quadruped_gym_env import QuadrupedGymEnv

parser = argparse.ArgumentParser()
parser.add_argument('--omega_swing', type=float, default=18) # usually 2-4 times of stance
parser.add_argument('--omega_stance', type=float, default=9)
parser.add_argument('--step_length', type=float, default=0.04)
parser.add_argument('--gait', type=str, default="TROT")
parser.add_argument('--record', dest='record', action='store_true')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--noise', dest='noise', action='store_true')
# parser.add_argument()
args = parser.parse_args()
logger.info(args)

# Fix random seed
np.random.seed(0)

class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.  

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**2,                # converge to sqrt(mu)
                omega_swing=args.omega_swing*2*np.pi,  # MUST EDIT
                omega_stance=args.omega_stance*2*np.pi, # MUST EDIT
                gait=args.gait,            # change depending on desired gait
                coupling_strength=args.omega_swing*2*np.pi,    # coefficient to multiply coupling matrix
                couple=True,            # should couple
                time_step=0.001,        # time step 
                ground_clearance=0.05,  # foot swing height 
                ground_penetration=0.01,# foot stance penetration into ground 
                robot_height=0.25,      # in nominal case (standing) 
                des_step_len=args.step_length,      # desired step length 
                ):
    
    ###############
    # initialize CPG data structures: amplitude is row 0, and phase is row 1
    self.X = np.zeros((2,4))

    # save parameters 
    self._mu = mu
    self._omega_swing = omega_swing
    self._omega_stance = omega_stance  
    self._couple = couple
    self._coupling_strength = coupling_strength
    self._dt = time_step
    self._set_gait(gait)

    # set oscillator initial conditions  
    self.X[0,:] = np.random.rand(4) * .1
    self.X[1,:] = self.PHI[0,:] 

    # save body and foot shaping
    self._ground_clearance = ground_clearance 
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height 
    self._des_step_len = des_step_len


  def _set_gait(self,gait):
    """ For coupling oscillators in phase space. 
    [TODO] update all coupling matrices
    """
    self.PHI_trot = np.array(
      ((0, np.pi, np.pi, 0),
       (-np.pi, 0, 0, -np.pi),
       (-np.pi, 0, 0, -np.pi),
       (0, np.pi, np.pi, 0))
    )
    # FR _xxxxxxx__
    # FL xxx___xxxx
    # RR xx___xxxxx
    # RL xxxxxxx___
    self.PHI_walk = 2*np.pi*np.array(
      ((0, 0.5, 0.75, 0.25),
       (-0.5, 0, 0.25, -0.25),
       (0.25, -0.25, 0, 0.5),
       (-0.25, 0.25, 0.5, 0))
    )
    # FR xxx_______ 
    # FL xxx_______
    # RR ____xxx___
    # RL ____xxx___
    self.PHI_bound = np.array(
      ((0, 0, np.pi, np.pi),
       (0, 0, np.pi, np.pi),
       (-np.pi, -np.pi,0, 0),
       (-np.pi, -np.pi,0, 0))
    )
    # FR xx__________
    # FL ___xx_______
    # RR ______xx____
    # RL _________xx_
    self.PHI_pace = np.array(
      ((0, np.pi, 0, np.pi),
       (np.pi, 0, np.pi, 0),
       (np.pi, 0, np.pi, 0),
       (0, np.pi, 0, np.pi))
    )

    if gait == "TROT":
      print('TROT')
      self.PHI = self.PHI_trot
    elif gait == "PACE":
      print('PACE')
      self.PHI = self.PHI_pace
    elif gait == "BOUND":
      print('BOUND')
      self.PHI = self.PHI_bound
    elif gait == "WALK":
      print('WALK')
      self.PHI = self.PHI_walk
    else:
      raise ValueError( gait + 'not implemented.')

  # @logger.catch
  def update(self):
    """ Update oscillator states. """

    # update parameters, integrate
    self._integrate_hopf_equations()
    
    # map CPG variables to Cartesian foot xz positions (Equations 8, 9) 
    r, theta = self.X[0, :], self.X[1, :]
    x = -self._des_step_len * r * np.cos(theta)
    z = np.zeros(4)
    for i in range(4):
      z[i] = -self._robot_height + self._ground_clearance * np.sin(theta[i]) if theta[i] < np.pi \
        else -self._robot_height + self._ground_penetration * np.sin(theta[i]) 
    return x, z
      
        
  def _integrate_hopf_equations(self):
    """ Hopf polar equations and integration. Use equations 6 and 7. """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    X_dot = np.zeros((2,4))
    alpha = 50 

    # loop through each leg's oscillator
    for i in range(4):
      # get r_i, theta_i from X
      r, theta = X[0, i], X[1, i] 
      # compute r_dot (Equation 6)
      r_dot = alpha * (self._mu - r**2) * r
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      omega = self._omega_swing if theta % 2*np.pi <= np.pi else self._omega_stance
      theta_dot = omega

      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        sum_rwsin = 0
        for j in range(4):
          r_j, theta_j = X[0, j], X[1, j]
          sum_rwsin += r_j * self._coupling_strength * np.sin(theta_j - theta - self.PHI[i, j])
        theta_dot += sum_rwsin

      # set X_dot[:,i]
      X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X += X_dot * self._dt
    # mod phase variables to keep between 0 and 2pi
    self.X[1,:] = self.X[1,:] % (2*np.pi)



if __name__ == "__main__":

  ADD_CARTESIAN_PD = True
  TIME_STEP = 0.001
  foot_y = 0.0838 # this is the hip length 
  sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

  env = QuadrupedGymEnv(render=True,              # visualize
                      on_rack=True,              # useful for debugging! 
                      isRLGymInterface=False,     # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="TORQUE",
                      add_noise=args.noise,    # start in ideal conditions
                      record_video=args.record,
                      )

  # initialize Hopf Network, supply gait
  cpg = HopfNetwork(time_step=TIME_STEP)

  TEST_STEPS = int(10 / (TIME_STEP))
  t = np.arange(TEST_STEPS)*TIME_STEP

  # [TODO] initialize data structures to save CPG and robot states
  joint_pos = []
  foot_pos = []

  ############## Sample Gains
  # joint PD gains
  kp=np.array([150,70,70])
  kd=np.array([2,0.5,0.5])
  # Cartesian PD gains
  kpCartesian = np.diag([2500]*3)
  kdCartesian = np.diag([40]*3)


  for j in range(TEST_STEPS):
    # initialize torque array to send to motors
    action = np.zeros(12) 
    # get desired foot positions from CPG 
    xs,zs = cpg.update()
    foot_pos.append([xs, zs])
    # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q = np.array(env.robot.GetMotorAngles()).reshape(4, -1) # shape [1, 12]
    dq = np.array(env.robot.GetMotorVelocities()).reshape(4, -1)

    # loop through desired foot positions and calculate torques
    for i in range(4):
      # initialize torques for legi
      tau = np.zeros(3)
      # get desired foot i pos (xi, yi, zi) in leg frame
      leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]]) # [1, 3]
      # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
      leg_q = np.array(env.robot.ComputeInverseKinematics(i, leg_xyz)) #[1, 3]
      # Add joint PD contribution to tau for leg i (Equation 4)
      tau += kp * (leg_q - q[i, :]) + kd * (-1*dq[i, :]) # [TODO] # [1, 3]
      # logger.debug(tau.shape)

      # add Cartesian PD contribution
      if ADD_CARTESIAN_PD:
        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
        J, pos = env.robot.ComputeJacobianAndPosition(i) # [1, 3]
        joint_pos.append(pos)
        # Get current foot velocity in leg frame (Equation 2)
        v = np.matmul(J, dq[i, :].transpose()) # {[3, 3] * [3, 1]}
        # logger.info(v.shape) 
        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        F = np.matmul(kpCartesian, (leg_xyz - pos).transpose()) + np.matmul(kdCartesian, (-v).transpose()) # [1, 3]
        tau += np.matmul(J.T, F)
        # logger.debug(tau.shape)
      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau

    # send torques to robot and simulate TIME_STEP seconds 
    env.step(action) 

    # [TODO] save any CPG or robot states
  foot_pos = np.array(foot_pos) # [-1, 2, 4]
  joint_pos = np.array(joint_pos).reshape(-1, 4, 3) # [-1, 4, 3]
  logger.info(foot_pos.shape)
  logger.info(joint_pos.shape)
  logger.info(t.shape)


  ##################################################### 
  # PLOTS
  #####################################################
  # example
  if args.plot:
    fig, ax = plt.subplots(2, 1, sharex=True)
    # ax.set_title('Plots of CPG')
    ax[0].plot(t, foot_pos[:, 1, 0], label='FR foot z')
    ax[0].set(title='Foot position')
    ax[1].plot(t, joint_pos[:, 0, 1], label='FR thigh')
    ax[1].set(title='Joint position')
    plt.show()