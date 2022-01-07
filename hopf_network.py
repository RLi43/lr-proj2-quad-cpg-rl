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

FOOT_Y = 0.0838
foot_y = FOOT_Y
foot_x = 0

class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.  

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**2,                # converge to sqrt(mu)
                # omega_swing=10*2*np.pi,  # TODO Swing Frequency
                # omega_stance=10*2*np.pi, # TODO Stance Frequency
                gait="WALK",            # change depending on desired gait
                # coupling_strength=1,    # coefficient to multiply coupling matrix
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

    # save body and foot shaping
    self._ground_clearance = ground_clearance 
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height 
    self._des_step_len = des_step_len

    # save parameters 
    self._mu = mu
    # self._omega_swing = omega_swing
    # self._omega_stance = omega_stance  
    self._couple = couple
    self._dt = time_step
    self._set_gait(gait) # _omega_swing, _omega_stance, _des_step_len, _ground_clearance, _ground_penetration
    self._coupling_strength = min(self._omega_swing,self._omega_stance)/3 * np.ones((4,4))

    # set oscillator initial conditions  
    self.X[0,:] = np.ones((1,4)) * .1 #
    self.X[1,:] = self.PHI[0,:] 

    # save r and theta
    self.X_dot = np.zeros((2,4))
    self.state = ["SWING",]*4


  def _set_gait(self,gait):
    """ For coupling oscillators in phase space. 
    [tODO] update all coupling matrices
    """
    # FL(2)  FR(1)
    # RL(4)  RR(3)

    def footfall2phi(footfall):
      ret = np.zeros((4,4))
      for i in range(4):
        for j in range(4):
          ret[i,j] = footfall[j] - footfall[i]
      return 2*np.pi*ret
    
    # 0.0    0.5
    # 0.5    0.0
    footfall_trot = [0.5, 0.0, 0.0, 0.5]    
    self.PHI_trot = footfall2phi(footfall_trot)

    # 0.5-     0.0-
    # 0.25     0.75
    footfall_walk = [0.0, 0.5, 0.75, 0.25]
    self.PHI_walk = footfall2phi(footfall_walk)
    # 0.5      0.0
    # 0.75     0.25
    footfall_walk_diagonal = [0.0, 0.5, 0.25, 0.75]
    self.PHI_walk_diagonal = footfall2phi(footfall_walk_diagonal)

    footfall_bound = [0.0, 0.0, 0.5, 0.5]
    self.PHI_bound = footfall2phi(footfall_bound)

    canter_param = 0.05 # suspension = 0.5 - 2*cp - duty(>0.25+cp)
    # 0.75+    0.5
    # 0.5      0.25-
    footfall_canter_transverse = [0.5, 0.75+canter_param, 0.25-canter_param, 0.5]
    self.PHI_canter_transverse = footfall2phi(footfall_canter_transverse)
    # 0.5      0.75+
    # 0.5      0.25-
    footfall_canter_rotatory = [0.75+canter_param, 0.5, 0.25-canter_param, 0.5]
    self.PHI_canter_rotatory = footfall2phi(footfall_canter_rotatory)

    gallop_param1 = 0.05
    gallop_param2 = 0.25 # suspension interval = 1 - (gp2+gp1) or 2x (1 - 2*gp2+gp1)/2
    if gait == "BOUND":
      gallop_param1 = 0
      gallop_param2 = 0.25
    elif gait == "GALLOP_TRANS" or gait == "GALLOP":
      gallop_param2 = 0.4
    # p      p+
    # 0      0+
    footfall_gallop_transverse = [gallop_param2+gallop_param1, gallop_param2, gallop_param1, 0]
    self.PHI_gallop_transverse = footfall2phi(footfall_gallop_transverse)
    # 2p      2p+
    # 0+      0
    footfall_gallop_rotatory = [2*gallop_param2+gallop_param1, 2*gallop_param2, 0, gallop_param1]
    self.PHI_gallop_rotatory = footfall2phi(footfall_gallop_rotatory)

    # 0.0      0.5
    # 0.0      0.5
    footfall_pace = [0.5, 0, 0.5, 0]
    self.PHI_pace = footfall2phi(footfall_pace)

    self.PHI_pronk = np.zeros((4,4))

    global foot_y, foot_x
    if gait == "WALK":
      print('WALK')
      self.PHI = self.PHI_walk
      # Very slow walk: 4-leg support: duty > 3/4
      self._omega_swing = 5.0*2*np.pi
      self._omega_stance = 1.0*2*np.pi
      # self._des_step_len = 0.04
      # self._ground_penetration = 0.01
      # self._ground_clearance = 0.05
    elif gait == "WALK_DIAG":
      print('WALK_DIAG')
      self.PHI = self.PHI_walk_diagonal
      # Normal walk: 3/2-leg support: 1/2 < duty < 3/4 => stance/swing in (1/3,1)
      self._omega_swing = 5.0*2*np.pi
      self._omega_stance = 3.0*2*np.pi
      # self._des_step_len = 0.04
      # self._ground_penetration = 0.01
      # self._ground_clearance = 0.05
    elif gait == "AMBLE":
      print(gait) 
      self.PHI = self.PHI_walk
      # fast amble: 2/1-leg support: 1/4 < duty < 1/2 => stance/swing in (1,3)
      self._omega_swing = 10.0*2*np.pi
      self._omega_stance = 15.0*2*np.pi
      # self._des_step_len = 0.04
      # self._ground_penetration = 0.01
      # self._ground_clearance = 0.05
    elif gait == "TROT" or gait == "TROT_RUN":
      print('TROT_RUN')
      self.PHI = self.PHI_trot
      # running trot: with suspension : duty < 1/2 => stance/swing > 1
      self._omega_swing = 9.0*2*np.pi
      self._omega_stance = 10*2*np.pi
      # self._des_step_len = 0.04
      # self._ground_penetration = 0.01
      # self._ground_clearance = 0.05
    elif gait == "TROT_WALK":
      print('TROT_WALK')
      self.PHI = self.PHI_trot
      # walking trot: without suspension : duty > 1/2 => stance/swing < 1
      self._omega_swing = 2.2*2*np.pi
      self._omega_stance = 2.0*2*np.pi
      # self._des_step_len = 0.04
      # self._ground_penetration = 0.01
      # self._ground_clearance = 0.05
    elif gait == "PACE":
      print('PACE')
      self.PHI = self.PHI_pace
      # walking-to-running pace
      # abnormal gait: the gravity point is shifting
      foot_y = FOOT_Y * 1.2
      self._robot_height *= 0.8
      # without suspension : duty >~ 1/2 => stance/swing ~< 1
      self._omega_swing = 8.0*2*np.pi
      self._omega_stance = 6.0*2*np.pi
      self._des_step_len = 0.05
      self._ground_penetration = 0.01
      self._ground_clearance = 0.06
    elif gait == "PACE_FLY":
      print('PACE_FLY') #[TODO]
      self.PHI = self.PHI_pace      
      # with suspension : duty < 1/2 => stance/swing > 1
      foot_y = FOOT_Y * 1.5
      self._robot_height *= 0.8
      self._omega_swing = 6.0*2*np.pi
      self._omega_stance = 9.0*2*np.pi
      self._des_step_len = 0.05
      #self._ground_penetration = 0.01
      #self._ground_clearance = 0.05
    elif gait == "CANTER_TRANS" or gait == "CANTER":
      print("CANTER_TRANS") # I will never change it anymore!
      self.PHI = self.PHI_canter_transverse
      foot_y = FOOT_Y * 1.2
      self._robot_height *= 0.8
      # ~ 0.35 ~ 13/7
      self._omega_swing = 6*2*np.pi
      self._omega_stance = 13*2*np.pi
      self._des_step_len = 0.07
      self._ground_penetration = 0.005
      self._ground_clearance = 0.03
    elif gait == "CANTER_ROTA":
      print("CANTER_ROTA")  # At least, it doesn't fall
      self.PHI = self.PHI_canter_rotatory
      # running
      foot_y = FOOT_Y * 1.2
      self._robot_height *= 0.8
      self._omega_swing = 6.0*2*np.pi
      self._omega_stance = 13.0*2*np.pi
      self._des_step_len = 0.07
      self._ground_penetration = 0.005
      self._ground_clearance = 0.06
    elif gait == "BOUND":
      print(gait)
      self.PHI = self.PHI_gallop_rotatory
      # running
      #foot_y = FOOT_Y * 1.2
      # 5.5, 10
      # ---------- At least not fall down for BOUND-----
      foot_x = 0.04
      foot_y = FOOT_Y * 1.2
      self._robot_height = 0.20
      self._omega_swing =  5.5*2*np.pi
      self._omega_stance = 22.0*2*np.pi
      self._des_step_len = 0.065
      self._ground_penetration = 0.02
      self._ground_clearance = 0.07
    elif gait == "GALLOP_ROTA":
      print(gait)
      self.PHI = self.PHI_gallop_rotatory
      # ---------- A walking dog  -----
      self._omega_swing =  11*2*np.pi
      self._omega_stance = 20.0*2*np.pi
      self._des_step_len = 0.05
      self._ground_penetration = 0.02
      self._ground_clearance = 0.05
      # ---------- It doesn't fall down in 10 secs  -----
      foot_y = FOOT_Y * 1.2
      self._robot_height = 0.23
      self._omega_swing =  9.5*2*np.pi
      self._omega_stance = 20.0*2*np.pi
      self._des_step_len = 0.05
      self._ground_penetration = 0.02
      self._ground_clearance = 0.06
    elif gait == "GALLOP_TRANS" or gait == "GALLOP":
      print(gait)
      self.PHI = self.PHI_gallop_transverse
      # running
      # foot_y = FOOT_Y * 1.2
      # self._robot_height *= 0.8
      # ---------- At least not fall down -----
      self._omega_swing = 10*2*np.pi
      self._omega_stance = 15.0*2*np.pi
      # ---------- Stable
      foot_y = FOOT_Y * 1.2
      self._robot_height *= 0.9
      self._omega_swing = 12*2*np.pi
      self._omega_stance = 20.0*2*np.pi
      self._des_step_len = 0.05
      self._ground_penetration = 0.02
      # self._ground_clearance = 0.05

      # ---------- Faster! -----
      # foot_x = 0.02
      # foot_y = FOOT_Y * 1.2
      # self._robot_height *= 0.9
      # self._omega_swing = 11*2*np.pi
      # self._omega_stance = 20.0*2*np.pi
      # self._des_step_len = 0.06
      # self._ground_penetration = 0.02
      # self._ground_clearance = 0.05
    elif gait == "PRONK":
      print('PRONK')
      self.PHI = self.PHI_pronk
      # ------- Stable ---------
      # self._robot_height *= 0.8
      self._omega_swing = 11*2*np.pi
      self._omega_stance = 25.0*2*np.pi
      self._des_step_len = 0.05
      self._ground_penetration = 0.01
      self._ground_clearance = 0.07
      # ------- Fast -- 3.50 --------
      self._robot_height = 0.20
      foot_y = FOOT_Y * 1.2
      foot_x = 0.02
      self._omega_swing = 11*2*np.pi
      self._omega_stance = 25.0*2*np.pi
      self._des_step_len = 0.065
      self._ground_penetration = 0.022
      self._ground_clearance = 0.07

    else:
      raise ValueError(gait + ' not implemented.')

  def update(self, body_ori = None, contactInfo = None):
    """ Update oscillator states. """

    # update parameters, integrate
    self._integrate_hopf_equations(body_ori, contactInfo)
    
    # map CPG variables to Cartesian foot xz positions (Equations 8, 9) 
    r, theta = self.X[0, :], self.X[1, :]
    if contactInfo is not None:
      contactBool, forceNormal = contactInfo
    x = -self._des_step_len * r * np.cos(theta) #[tODO]
    z = np.zeros(4)
    factor = 1.0
    for i in range(4):
      if theta[i] < np.pi:
        g = self._ground_clearance 
        self.state[i] = "SWING"
        # if contactInfo is not None and contactBool[i]:
        #   factor = 0.9
      else:
        g = self._ground_penetration
        self.state[i] = "STANCE"
        # if contactInfo is not None and not contactBool[i]:
        #   factor = 1.1
      # if i < 2:
      #   factor = 1.25
      # else: factor = 0.75
      z[i] = -self._robot_height + r[i]*np.sin(theta[i]) * g *factor# [tODO]
      
    # if body_ori is not None:
    #   roll, pitch, yaw = body_ori
        

    return x, z
      
        
  def _integrate_hopf_equations(self, body_ori = None, contactInfo = None):
    """ Hopf polar equations and integration. Use equations 6 and 7. """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    self.X_dot = np.zeros((2,4))
    alpha = 5.0 
    F = 10
    force_threshold = 10 # 120N/4

    # loop through each leg's oscillator
    for i in range(4):
      # get r_i, theta_i from X
      r, theta = X[0,i], X[1,i] # [tODO]
      # compute r_dot (Equation 6)
      r_dot = alpha * (self._mu - r**2) * r # [tODO]
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      # swinging
      if theta < np.pi:
        theta_dot = self._omega_swing
      else:
        theta_dot = self._omega_stance
      
      #theta_dot *= np.cos(theta) + 1.1 #abs(0.3+0.7*np.pi/2*np.sin(theta))

      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        for j in range(4):
          delta_theta = (X[1, j] - theta - self.PHI[i, j]) #% 2*np.pi
          # delta_theta = min(0.5*np.pi, delta_theta) 
          # if delta_theta < 0:
          #   delta_theta = 0
          theta_dot += X[0, j] * self._coupling_strength[i,j] * np.sin(delta_theta) # [tODO] Question?
      
      if theta_dot < 0:
        theta_dot = 0

      if contactInfo is not None:
        contactBool, forceNormal = contactInfo
        # [2] III B
        # Stopping before transition
        # 1. swing to stance, not contact
        # 2. stance to swing, support a lot
        # Fast transitions
        # 1. during stance, support few
        # 2. during swing, contact
        suppose_theta = theta + theta_dot*self._dt*1.1 #
        if theta < np.pi: #swing
          if suppose_theta > np.pi and not contactBool[i]:
            theta_dot = 0
          elif contactBool[i]:
            theta_dot *= 2.0
        else:
          if suppose_theta > 2*np.pi and (forceNormal[i] > force_threshold): # TODO
            theta_dot = 0
          elif forceNormal[i] < force_threshold:
            theta_dot *= 2.0

      # set X_dot[:,i]
      self.X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X += self.X_dot*self._dt  # [tODO]
    # mod phase variables to keep between 0 and 2pi
    self.X[1,:] = self.X[1,:] % (2*np.pi)


if __name__ == "__main__":

  ADD_CARTESIAN_PD = True
  ADD_JOINT_PD = True
  USE_FEEDBACK = False
  ON_RACK = False
  TRACK_DIRECTION = False
  gait_direction = 0 #np.pi # forward
  gait_name = "WALK"
  simulation_time = 10.0
  PLAY_SPEED = 0.25

  if not ADD_CARTESIAN_PD and not ADD_JOINT_PD:
    raise("At least one PD needed")
  if ON_RACK and USE_FEEDBACK:
    raise("Can not use feedback when on rack")
  notpureforward = True if gait_direction != 0.0 else False
  TIME_STEP = 0.001
  #foot_y = 0.0838 # this is the hip length 
  sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

  env = QuadrupedGymEnv(render=True,              # visualize
                      on_rack=ON_RACK,              # useful for debugging! 
                      isRLGymInterface=False,     # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="PD",
                      add_noise=False,    # start in ideal conditions
                      #record_video=True
                      )
  # env.add_random_boxes()


  # initialize Hopf Network, supply gait
  cpg = HopfNetwork(time_step=TIME_STEP, gait=gait_name,
                ground_clearance=0.05,  # foot swing height 
                ground_penetration=0.01)# foot stance penetration into ground )

  TEST_STEPS = int(simulation_time / (TIME_STEP))
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
  orientation_history = np.zeros((TEST_STEPS, 3))
  contact_history = np.zeros((TEST_STEPS, 4))
  speed_history = np.zeros((TEST_STEPS, 3))

  ############## Sample Gains
  # joint PD gains
  kp=np.array([150,70,70])
  kd=np.array([2,0.5,0.5])
  # Cartesian PD gains
  kpCartesian = np.diag([2500]*3)
  kdCartesian = np.diag([40]*3)

  total_starter = time.time()

  for j in range(TEST_STEPS):
    starter = time.time()
    # initialize torque array to send to motors
    action = np.zeros(12) 
    # get desired foot positions from CPG 
    roll, pitch, yaw = env.robot.GetBaseOrientationRollPitchYaw()
    #yaw = None
    orientation_history[j] = np.array((roll, pitch, yaw))
    numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
    if USE_FEEDBACK:
      xs,zs = cpg.update(body_ori = (roll, pitch, yaw), contactInfo=(feetInContactBool, feetNormalForces)) #feetInContactBool
    else:
      xs,zs = cpg.update(body_ori = (roll, pitch, yaw))
    ys = foot_y * sideSign
    if notpureforward or TRACK_DIRECTION:
      forward_direction = gait_direction-yaw
      ys += np.sin(forward_direction) * xs
      xs = np.cos(forward_direction) * xs

    xs += foot_x
    if False:
      if roll > 0:
        ys[0] *= 1 + roll/np.pi/2
        ys[2] *= 1 + roll/np.pi/2
      else:
        ys[1] *= 1 + roll/np.pi/2
        ys[3] *= 1 + roll/np.pi/2

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
      leg_xyz = np.array([xs[i],ys[i],zs[i]])
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
    contact_history[j,:] = 2.0*np.array(feetInContactBool)
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

    linear_vel = env.robot.GetBaseLinearVelocity()
    speed_history[j, :] = linear_vel
    loop_time = time.time() - total_starter
    if env._is_render:
      if loop_time < TIME_STEP/PLAY_SPEED*(j+1):
      # print("sleep", TIME_STEP - loop_time)
        time.sleep(TIME_STEP/PLAY_SPEED*(j+1) - loop_time)
      # else:
      #   print("\rovertime", loop_time - TIME_STEP/PLAY_SPEED*(j+1), end="")

  cur_position = env.robot.GetBasePosition()
  distance_traveled = np.linalg.norm(cur_position)
  env.close()


  ##################################################### 
  # PLOTS
  #####################################################
  print(enery_cost)
  enery_cost = sum(enery_cost)
  print("Energy Cost:", enery_cost)
  print("Distance Traveled:", distance_traveled)
  print("Cost of Transport:", enery_cost/distance_traveled)
  print("average abs(pitch):", np.mean(abs(orientation_history[:,1])))
  print("orientation variance:", np.var(orientation_history, axis=0))
  print("average x-y speed(last50%)", np.mean(np.linalg.norm(speed_history[int(len(t)/2):, :2], axis=1)))

  # fig = plt.figure()
  # plt.plot(t, action_history)
  # example
  fig = plt.figure()
  leg_names = ["FR", "FL", "RR", "RL"]
  scales = [1, np.pi, 10, 100]
  for i in range(4):
    plt.subplot(4,1,1+i)
    plt.title(leg_names[i])
    for j in range(4):
      plt.plot(t, cpg_history[:,j,i]/scales[j])
    #plt.plot(t, contact_history[:, i])
    plt.legend(["r","theta","dr","dtheta"])#,"contact"])
  #plt.subplot(5,1,5)
  #plt.plot(t, orientation_history)
  #plt.legend(["roll","pitch","yaw"])
  plt.tight_layout()

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
    start_of_a_cycle = history_phase_change[:-1].reshape(-1, 2)
  except:
    start_of_a_cycle = history_phase_change[:-2].reshape(-1, 2)

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
  plt.plot(start_of_a_cycle, history_duration[:, 0]*TIME_STEP)
  plt.plot(start_of_a_cycle, history_duration[:, 1]*TIME_STEP)
  plt.legend(["Swing Phase Duration", "Stance Phase Duration"])

  fig = plt.figure()
  plt.subplot(2,1,1)
  plt.plot(t, speed_history)
  plt.grid()
  plt.title("linear speed")
  plt.legend(["x", "y", "z"])
  plt.subplot(2,1,2)
  plt.title("x-y speed")
  plt.plot(t, np.linalg.norm(speed_history[:, :2], axis=1))
  plt.grid()

  print("duty factor", duty_factors[-1])
  print("swing duration", history_duration[-1, 0]*TIME_STEP)
  print("stance duration", history_duration[-1, 1]*TIME_STEP)

  plt.show()