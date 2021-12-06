"""
CPG in polar coordinates based on: 
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import time
import argparse
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
                omega_swing=5*2*np.pi,  # MUST EDIT
                omega_stance=2*2*np.pi, # MUST EDIT
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
    self.PHI_trot = np.array([[0, -np.pi, -np.pi, 0], [np.pi, 0, 0, np.pi], [np.pi, 0, 0, np.pi], [0, -np.pi, -np.pi, 0]])
    self.PHI_walk = np.zeros((4,4))
    self.PHI_bound = np.zeros((4,4))
    self.PHI_pace = np.array([[0, 1/2*np.pi, np.pi, 3/2*np.pi], [-1/2*np.pi, 0, 1/2*np.pi, np.pi], [-np.pi, -1/2*np.pi, 0, 1/2*np.pi], [-3/2*np.pi, -np.pi, -1/2*np.pi, 0]])

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
    x = -self._des_step_len*self.X[0, :]*np.cos(self.X[1, :])
    indicator = np.int64(np.sin(self.X[1, :]) > 0)
    z = -self._robot_height + (self._ground_clearance*indicator + self._ground_penetration*(1 - indicator))*np.sin(self.X[1, :])

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
      r, theta = X[:, i]
      # compute r_dot (Equation 6)
      r_dot = alpha*(self._mu - r**2)*r
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      indicator = int(np.sin(theta) > 0)
      theta_dot = self._omega_swing*indicator + self._omega_stance*(1 - indicator)

      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        theta_dot += np.dot(self._coupling_strength*X[0, :], np.sin(X[1, :] - theta - self.PHI[i, :]))

      # set X_dot[:,i]
      X_dot[:,i] = [r_dot, theta_dot]
    
    # save X_dot for plotting
    self.X_dot = X_dot.copy()

    # integrate 
    self.X = X + X_dot*self._dt
    # mod phase variables to keep between 0 and 2pi
    self.X[1,:] = self.X[1,:] % (2*np.pi)


def parse_arguments():
    parser = argparse.ArgumentParser(description='CPG Arguments Parser')
    parser.add_argument('--gait', type=str, default='TROT', help='TROT, PACE, BOUND or WALK')
    parser.add_argument('-s', '--store', action='store_true', help='Store CPG and robot states')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot CPG and robot states')
    parser.add_argument('-v', '--video', action='store_true', help='Record a video')

    return parser.parse_args()


if __name__ == "__main__":
  args = parse_arguments()

  ADD_CARTESIAN_PD = True
  TIME_STEP = 0.001
  foot_y = 0.0838 # this is the hip length 
  sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

  env = QuadrupedGymEnv(render=True,              # visualize
                      on_rack=False,              # useful for debugging! 
                      isRLGymInterface=False,     # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="TORQUE",
                      add_noise=False,    # start in ideal conditions
                      record_video=args.video
                      )

  # initialize Hopf Network, supply gait
  cpg = HopfNetwork(time_step=TIME_STEP, gait=args.gait)

  TEST_STEPS = int(10 / (TIME_STEP))
  t = np.arange(TEST_STEPS)*TIME_STEP

  # [TODO] initialize data structures to save CPG and robot states
  # CPG = np.zeros((TEST_STEPS, 4, 2))
  # Pos_d = np.zeros((TEST_STEPS, 4, 3))
  # Pos = np.zeros((TEST_STEPS, 4, 3))
  # Joint_d = np.zeros((TEST_STEPS, 4, 3))
  # Joint = np.zeros((TEST_STEPS, 4, 3))

  # States by row: r, theta, r_dot, theta_dot, posd(x, y, z), pos, qd, q
  States = np.zeros((TEST_STEPS, 16, 4))


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
    # get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q = np.transpose(env.robot.GetMotorAngles().reshape(4, 3))
    dq = np.transpose(env.robot.GetMotorVelocities().reshape(4, 3))

    pos_d = np.zeros((3, 4))
    pos = np.zeros((3, 4))
    joint_d = np.zeros((3, 4))

    # loop through desired foot positions and calculate torques
    for i in range(4):
      # initialize torques for legi
      tau = np.zeros(3)
      # get desired foot i pos (xi, yi, zi) in leg frame
      leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
      # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
      leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)
      # Add joint PD contribution to tau for leg i (Equation 4)
      leg_dq = (leg_q - q[:, i])/TIME_STEP*0.01
      tau += kp*(leg_q - q[:, i]) + kd*(leg_dq - dq[:, i])
      # tau += kp*(leg_q - q[:, i]) + kd*(-dq[:, i])

      # add Cartesian PD contribution
      if ADD_CARTESIAN_PD:
        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
        Jacobian, p = env.robot.ComputeJacobianAndPosition(i)
        # Get current foot velocity in leg frame (Equation 2)
        v = Jacobian@dq[:, i]
        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        leg_dxyz = (leg_xyz - p)/TIME_STEP*0.01
        tau += np.transpose(Jacobian)@(kpCartesian@(leg_xyz - p) + kdCartesian@(leg_dxyz - v))
        # tau += np.transpose(Jacobian)@(kpCartesian@(leg_xyz - p) + kdCartesian@(-v))

      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau

      if args.store or args.plot:
        pos_d[:, i] = leg_xyz
        pos[:, i] = p
        joint_d[:, i] = leg_q

    # send torques to robot and simulate TIME_STEP seconds 
    env.step(action) 

    # save any CPG or robot states
    if args.store or args.plot:
      States[j, 0:2, :] = cpg.X
      States[j, 2:4, :] = cpg.X_dot
      States[j, 4:7, :] = pos_d
      States[j, 7:10, :] = pos
      States[j, 10:13, :] = joint_d
      States[j, 13:16, :] = q
  
  if args.store:
    ofile = open('robot_states.csv', 'wb')
    np.savetxt(ofile, States.reshape(TEST_STEPS, -1), delimiter=',')
    ofile.close()


  ##################################################### 
  # PLOTS
  #####################################################
  if args.plot:
    leg_name = ['FR', 'FL', 'RR', 'RL']
    coord_name = ['x', 'y', 'z']
    joint_name = ['hip', 'thigh', 'calf']

    # CPG
    fig, axs = plt.subplots(4, 1)
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()

    for i in range(4):
      axs[i].plot(t, States[:, 0, i], label=f'{leg_name[i]} r')
      axs[i].plot(t, States[:, 1, i], label=f'{leg_name[i]} theta')
      axs[i].plot(t, States[:, 2, i], label=f'{leg_name[i]} r_dot')
      axs[i].plot(t, States[:, 3, i], label=f'{leg_name[i]} theta_dot')
      axs[i].set_title(leg_name[i])
      axs[i].legend()

    plt.show()

    # pos_d vs. pos
    fig, axs = plt.subplots(3, 1)
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()

    leg_no = 0

    for i in range(3):
      axs[i].plot(t, States[:, 4+i, leg_no], label=f'{leg_name[leg_no]} {coord_name[i]} desired')
      axs[i].plot(t, States[:, 7+i, leg_no], label=f'{leg_name[leg_no]} {coord_name[i]}')
      axs[i].set_title(coord_name[i])
      axs[i].legend()
    
    plt.show()

    # joint_d vs. joint
    fig, axs = plt.subplots(3, 1)
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()

    leg_no = 0

    for i in range(3):
      axs[i].plot(t, States[:, 10+i, leg_no], label=f'{leg_name[leg_no]} {joint_name[i]} desired')
      axs[i].plot(t, States[:, 13+i, leg_no], label=f'{leg_name[leg_no]} {joint_name[i]}')
      axs[i].set_title(joint_name[i])
      axs[i].legend()
    
    plt.show()
    