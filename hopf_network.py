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
                omega = 10*2*np.pi,
                # omega_swing=10*2*np.pi,  # TODO Swing Frequency
                # omega_stance=10*2*np.pi, # TODO Stance Frequency
                gait="WALK",            # change depending on desired gait
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
    self._omega = omega
    # self._omega_swing = omega_swing
    # self._omega_stance = omega_stance  
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

    # save r and theta
    self.X_dot = np.zeros((2,4))
    self.state = ["SWING",]*4


  def _set_gait(self,gait):
    """ For coupling oscillators in phase space. 
    [tODO] update all coupling matrices
    """
    # FL(2)  FR(1)
    # RL(4)  RR(3)
    
    # 0.0    0.5
    # 0.5    0.0
    self.PHI_trot = 2*np.pi*np.array(
      ((0, 0.5, 0.5, 0),
       (-0.5, 0, 0, -0.5),
       (-0.5, 0, 0, -0.5),
       (0, 0.5, 0.5, 0))
    )
    
    # 0.5      0.0
    # 0.25     0.75
    self.PHI_walk = 2*np.pi*np.array(
      ((0, 0.5, 0.75, 0.25),
       (-0.5, 0, 0.25, -0.25),
       (0.25, -0.25, 0, 0.5),
       (-0.25, 0.25, 0.5, 0))
    )
    
    # 0.5      0.5
    # 0.0      0.0
    self.PHI_bound = np.array(
      ((0, 0, np.pi, np.pi),
       (0, 0, np.pi, np.pi),
       (-np.pi, -np.pi,0, 0),
       (-np.pi, -np.pi,0, 0))
    )

    # 0.0      0.5
    # 0.0      0.5
    self.PHI_pace = np.array(
      ((0, np.pi, 0, np.pi),
       (np.pi, 0, np.pi, 0),
       (np.pi, 0, np.pi, 0),
       (0, np.pi, 0, np.pi))
    )

    if gait == "TROT":
      print('TROT')
      self.PHI = self.PHI_trot
      # running trot
      self._omega_swing = self._omega
      self._omega_stance = self._omega * 4.0
    elif gait == "PACE":
      print('PACE')
      self.PHI = self.PHI_pace
      # running pace
      self._omega_swing = self._omega
      self._omega_stance = self._omega * 4.0
    elif gait == "BOUND":
      print('BOUND')
      self.PHI = self.PHI_bound
      # running bound
      self._omega_swing = self._omega
      self._omega_stance = self._omega * 4.0
    elif gait == "WALK":
      print('WALK')
      self.PHI = self.PHI_walk
      self._omega_swing = self._omega * 2.0
      self._omega_stance = self._omega
    else:
      raise ValueError( gait + 'not implemented.')


  def update(self, contactInfo = None):
    """ Update oscillator states. """

    # update parameters, integrate
    self._integrate_hopf_equations(contactInfo)
    
    # map CPG variables to Cartesian foot xz positions (Equations 8, 9) 
    r, theta = self.X[0, :], self.X[1, :]
    x = -self._des_step_len * r * np.cos(theta) #[tODO]
    z = np.zeros(4)
    for i in range(4):
      if theta[i] < np.pi:
        g = self._ground_clearance 
        self.state[i] = "SWING"
      else:
        g = self._ground_penetration
        self.state[i] = "STANCE"
      z[i] = -self._robot_height + np.sin(self.X[1, i]) * g # [tODO]

    return x, z
      
        
  def _integrate_hopf_equations(self, contactBool = None):
    """ Hopf polar equations and integration. Use equations 6 and 7. """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    self.X_dot = np.zeros((2,4))
    alpha = 50 
    F = 10

    # loop through each leg's oscillator
    for i in range(4):
      # get r_i, theta_i from X
      r, theta = X[0,i], X[1,i] # [tODO]
      # compute r_dot (Equation 6)
      r_dot = alpha * (self._mu - r**2) * r # [tODO]
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      # swinging
      if contactBool is None:
        if theta < np.pi:
          theta_dot = self._omega_swing
        else:
          theta_dot = self._omega_stance
      else:
        if theta < np.pi:
          theta_dot = self._omega_swing + F
        else:
          theta_dot = self._omega_stance + F
      # theta_dot = self._omega_swing if theta < np.pi else self._omega_stance # [tODO]

      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        for j in range(4):
          theta_dot += X[0, j] * self._coupling_strength * np.sin(X[1, j] - theta - self.PHI[i, j]) # [tODO] Question?

      # set X_dot[:,i]
      self.X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X += self.X_dot*self._dt  # [tODO]
    # mod phase variables to keep between 0 and 2pi
    self.X[1,:] = self.X[1,:] % (2*np.pi)



if __name__ == "__main__":

  ADD_CARTESIAN_PD = True
  ADD_JOINT_PD = False
  if not ADD_CARTESIAN_PD and not ADD_JOINT_PD:
    raise("At least one PD needed")
  TIME_STEP = 0.001
  foot_y = 0.0838 # this is the hip length 
  sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

  env = QuadrupedGymEnv(render=True,              # visualize
                      on_rack=False,              # useful for debugging! 
                      isRLGymInterface=False,     # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="PD",
                      add_noise=False,    # start in ideal conditions
                      # record_video=True
                      )

  # initialize Hopf Network, supply gait
  cpg = HopfNetwork(time_step=TIME_STEP, gait="TROT", omega=2*np.pi)

  TEST_STEPS = int(3 / (TIME_STEP))
  t = np.arange(TEST_STEPS)*TIME_STEP

  # initialize data structures to save CPG and robot states
  # No. step * 4 legs * (r, theta, dr, dtheta)
  cpg_history = np.zeros((TEST_STEPS, 4, 4))
  action_history = np.zeros((TEST_STEPS, 12))
  history_leg_ind = 0
  history_leg_motor_ids = [i*4+history_leg_ind for i in range(3)]
  foot_desire_history = np.zeros((TEST_STEPS, 3))
  foot_real_history = np.zeros((TEST_STEPS, 3))
  angles_desire_history = np.zeros((TEST_STEPS, 3))
  angles_real_history = np.zeros((TEST_STEPS, 3))
  velocity_history = np.zeros((TEST_STEPS, 3))
  history_phase_change = [0]
  last_state = cpg.state[history_leg_ind]
  enery_cost = 0.0

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
    numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
    xs,zs = cpg.update() #feetInContactBool
    # [tODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q_last = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()
    q = np.array([q_last]).reshape(4, -1)
    dq = np.array([dq]).reshape(4, -1)

    # loop through desired foot positions and calculate torques
    for i in range(4):
      # initialize torques for legi
      tau = np.zeros(3)
      # get desired foot i pos (xi, yi, zi) in leg frame
      leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
      # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
      leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) # [tODO] # Coordination: Shoulder

      # Joint PD
      if ADD_JOINT_PD:
        # Add joint PD contribution to tau for leg i (Equation 4)
        tau += kp * (leg_q - q[i]) + kd * -dq[i] # [tODO]  

      # add Cartesian PD contribution
      if ADD_CARTESIAN_PD:
        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
        J, pos = env.robot.ComputeJacobianAndPosition(i)
        # Get current foot velocity in leg frame (Equation 2)
        vel = np.matmul(J, np.array([dq[i]]).T)
        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        tau += np.matmul(J.T, np.matmul(kpCartesian,np.array([leg_xyz - pos]).T) + np.matmul(kdCartesian, -vel))[:,0] # [tODO]

      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau

      if i == history_leg_ind:
        foot_desire_history[j,:] = leg_xyz
        angles_desire_history[j,:] = leg_q

    # send torques to robot and simulate TIME_STEP seconds 
    env.step(action) 
    # print(F"[{env.get_sim_time():.3f}] velocity = {np.linalg.norm(env.robot.GetBaseLinearVelocity()):.2f}")

    # [tODO] save any CPG or robot states
    cpg_history[j,:] = np.concatenate((cpg.X, cpg.X_dot), axis=0)
    action_history[j, :] = action
    J, foot_real_history[j, :] = env.robot.ComputeJacobianAndPosition(history_leg_ind)
    q_update = env.robot.GetMotorAngles()
    angles_real_history[j,:] = q_update[history_leg_motor_ids]
    velocity_history[j,:] = env.robot.GetBaseLinearVelocity()
    cur_state = cpg.state[history_leg_ind]
    if cur_state != last_state:
      last_state = cur_state
      history_phase_change.append(j)
    enery_cost += (q_update - q_last) * action




  ##################################################### 
  # PLOTS
  #####################################################
  print(enery_cost)
  enery_cost = sum(enery_cost)
  print("Energy Cost:", enery_cost)
  cur_position = env.robot.GetBasePosition()
  distance_traveled = np.linalg.norm(cur_position)
  print("Distance Traveled:", distance_traveled)
  print("Cost of Transport:", enery_cost/distance_traveled)
  # example
  fig = plt.figure()
  leg_names = ["FR", "FL", "RR", "RL"]
  scales = [1, np.pi, 10, 100]
  for i in range(4):
    plt.subplot(4,1,1+i)
    plt.title(leg_names[i])
    for j in range(4):
      plt.plot(t, cpg_history[:,j,i]/scales[j])
    plt.legend(["r","theta","dr","dtheta"])

  fig = plt.figure()
  plt.title("Position " + leg_names[history_leg_ind] +" Real vs Desire")
  plt.plot(t, foot_real_history)
  plt.plot(t, foot_desire_history)
  plt.legend(["real_x","real_y","real_z","desire_x","desire_y","desire_z"])

  fig = plt.figure()
  plt.title("Angles " + leg_names[history_leg_ind] +" Real vs Desire")
  plt.plot(t, angles_real_history)
  plt.plot(t, angles_desire_history)
  plt.legend(["real_thigh","real_calf","real_foot","desire_thigh","desire_calf","desire_foot"])

  history_phase_change = np.array(history_phase_change)
  history_duration = history_phase_change[1:] - history_phase_change[:-1]
  try:
    start_of_a_cycle = history_phase_change.reshape(-1, 2)
  except:
    start_of_a_cycle = history_phase_change[:-1].reshape(-1, 2)

  start_of_a_cycle = start_of_a_cycle[:,0]*TIME_STEP
  k = int(len(history_duration)/2)
  history_duration = np.array(history_duration[:2*k]).reshape(-1, 2)
  duty_factors = history_duration[:, 1] / np.sum(history_duration, axis = 1)
  fig = plt.figure()
  plt.subplot(2,1,1)
  plt.title("Duty Factor")
  plt.plot(start_of_a_cycle, duty_factors)
  plt.subplot(2,1,2)
  plt.title("Phase Durations")
  plt.plot(start_of_a_cycle, history_duration[:, 0])
  plt.plot(start_of_a_cycle, history_duration[:, 1])
  plt.legend(["Swing Phase Duration", "Stance Phase Duration"])

  plt.show()

  # plt.plot(action_history[:, 0], label = 'action FR q0')
  # plt.plot(action_history[:, 3], label = 'action FL q0')
  # plt.show()