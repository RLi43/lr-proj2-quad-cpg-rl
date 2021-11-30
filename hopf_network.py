"""
CPG in polar coordinates based on: 
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import time
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


class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.  

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**2,                # converge to sqrt(mu)
                omega_swing=1*2*np.pi,  # TODO Swing Frequency
                omega_stance=1*2*np.pi, # TODO Stance Frequency
                gait="TROT",            # change depending on desired gait
                coupling_strength=1,    # coefficient to multiply coupling matrix
                couple=True,            # should couple
                time_step=0.001,        # time step 
                ground_clearance=0.05,  # foot swing height 
                ground_penetration=0.01,# foot stance penetration into ground 
                robot_height=0.25,      # in nominal case (standing) 
                des_step_len=0.04,      # desired step length 
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
    self.X[0,:] = np.random.rand(4) * .1 #
    self.X[1,:] = self.PHI[0,:] 

    # save body and foot shaping
    self._ground_clearance = ground_clearance 
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height 
    self._des_step_len = des_step_len


  def _set_gait(self,gait):
    """ For coupling oscillators in phase space. 
    [tODO] update all coupling matrices
    """
    
    # FR xxxx______ # add x to walking trot
    # FL _____xxxx_
    # RR _____xxxx_
    # RL xxxx______
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
    p = 0.4
    self.PHI_walk = np.array(
      ((0, p*np.pi, np.pi, (1+p)*np.pi),
       (-p*np.pi, 0, (1-p)*np.pi, np.pi),
       (-np.pi, (1-p)*np.pi, 0, p*np.pi),
       ((-1-p)*np.pi, -np.pi, -p*np.pi, 0))
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
      ((0, 0.5*np.pi, np.pi, 1.5*np.pi),
       (-0.5*np.pi, 0, 0.5*np.pi, np.pi),
       (-np.pi, -0.5*np.pi, 0, 0.5*np.pi),
       (-1.5*np.pi, -np.pi, -0.5*np.pi, 0))
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


  def update(self):
    """ Update oscillator states. """

    # update parameters, integrate
    self._integrate_hopf_equations()
    
    # map CPG variables to Cartesian foot xz positions (Equations 8, 9) 
    x = -self._des_step_len * self.X[0, :] * np.cos(self.X[1, :]) # [tODO]
    z = np.zeros(4)
    for i in range(4):
      g = self._ground_clearance if self.X[0, i] < np.pi else self._ground_penetration
      z[i] = -self._robot_height + np.sin(self.X[1, i]) * g # [tODO]

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
      r, theta = X[0,i], X[1,i] # [tODO]
      # compute r_dot (Equation 6)
      r_dot = alpha * (self._mu - r**2) * r # [tODO]
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      theta_dot = self._omega_swing if theta < np.pi else self._omega_stance # [tODO]

      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        for j in range(4):
          theta_dot += X[0, j] * self._coupling_strength * np.sin(X[1, j] - theta - self.PHI[i, j]) # [tODO] Question?

      # set X_dot[:,i]
      X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X += X_dot*self._dt  # [tODO]
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
                      add_noise=False,    # start in ideal conditions
                      # record_video=True
                      )

  # initialize Hopf Network, supply gait
  cpg = HopfNetwork(time_step=TIME_STEP)

  TEST_STEPS = int(1 / (TIME_STEP))
  t = np.arange(TEST_STEPS)*TIME_STEP

  # [TODO] initialize data structures to save CPG and robot states
  cpg_history = np.zeros((TEST_STEPS, 2, 4))
  action_history = np.zeros((TEST_STEPS, 12))

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
    # [tODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()
    dq = np.array([dq]).reshape(4, -1)

    # loop through desired foot positions and calculate torques
    for i in range(4):
      # initialize torques for legi
      tau = np.zeros(3)
      # get desired foot i pos (xi, yi, zi) in leg frame
      leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
      # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
      leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) # [TODO] # Coordination: Shoulder
      # Add joint PD contribution to tau for leg i (Equation 4)
      tau += kp * (leg_q - q[i]) + kd * -dq[i] # [TODO]  

      # add Cartesian PD contribution
      if ADD_CARTESIAN_PD:
        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
        J, pos = env.robot.ComputeJacobianAndPosition(i)
        # Get current foot velocity in leg frame (Equation 2)
        vel = np.matmul(J, np.array([dq[i]]).T)
        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        tau += np.matmul(J.T, np.matmul(kpCartesian,np.array([leg_xyz - pos]).T) + np.matmul(kdCartesian, -vel))[:,0] # [TODO]

      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau

    # send torques to robot and simulate TIME_STEP seconds 
    env.step(action) 

    # [TODO] save any CPG or robot states
    cpg_history[j,:] = np.array((xs, zs))
    action_history[j, :] = action



  ##################################################### 
  # PLOTS
  #####################################################
  # example
  fig = plt.figure()
  plt.plot(t,cpg_history[:,0, 1], label='cpg xs FL')
  plt.plot(t,cpg_history[:,1, 1], label='cpg zs FL')
  plt.legend()
  plt.show()

  plt.plot(action_history[:, 0], label = 'action FR q0')
  plt.plot(action_history[:, 3], label = 'action FL q0')
  plt.show()