"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----


"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import do_mpc
from casadi import *
from scipy.spatial.transform import Rotation as R

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 100
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    start_pos = np.array([[0, 0, 0]])  
    start_orient = np.array([[0, 0, 0]]) 

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=start_pos,
                        initial_rpys=start_orient,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    ctrl = DSLPIDControl(drone_model=drone)

    # Initialize do-MPC
    model = do_mpc.model.Model('discrete')
    
    # Define constants
    grav_vector = np.array([0, 0, -9.8]).T
    m = ctrl._getURDFParameter('m')
    ct = ctrl._getURDFParameter('kf')
    cq = ctrl._getURDFParameter('kn')
    l = ctrl._getURDFParameter('arm')
    dt = 1/control_freq_hz

    # Define states (state vector x)
    x = model.set_variable(var_type='_x', var_name='x')
    y = model.set_variable(var_type='_x', var_name='y')
    z = model.set_variable(var_type='_x', var_name='z')
    x_dot = model.set_variable(var_type='_x', var_name='x_dot')
    y_dot = model.set_variable(var_type='_x', var_name='y_dot')
    z_dot = model.set_variable(var_type='_x', var_name='z_dot')
    q_w = model.set_variable(var_type='_x', var_name='q_w')
    q_x = model.set_variable(var_type='_x', var_name='q_x')
    q_y = model.set_variable(var_type='_x', var_name='q_y')
    q_z = model.set_variable(var_type='_x', var_name='q_z')
    omega_x = model.set_variable(var_type='_x', var_name='omega_x')
    omega_y = model.set_variable(var_type='_x', var_name='omega_y')
    omega_z = model.set_variable(var_type='_x', var_name='omega_z')

    G = np.array([[1, 1, 1, 1],
                  [l * np.sqrt(2)/2, -l * np.sqrt(2)/2, -l * np.sqrt(2)/2, l * np.sqrt(2)/2],
                  [-l * np.sqrt(2)/2, -l * np.sqrt(2)/2, l * np.sqrt(2)/2, l * np.sqrt(2)/2],
                  [cq/ct, -cq/ct, cq/ct, -cq/ct]])

    T = model.set_variable(var_type='_u', var_name='T')

    # Define rotor speeds (omega_1, omega_2, omega_3, omega_4)
    omega_1 = model.set_variable(var_type='_u', var_name='omega_1')
    omega_2 = model.set_variable(var_type='_u', var_name='omega_2')
    omega_3 = model.set_variable(var_type='_u', var_name='omega_3')
    omega_4 = model.set_variable(var_type='_u', var_name='omega_4')

    # Define thrust forces u as a function of rotor speeds
    u1 = model.set_variable(var_type='_z', var_name='u1')
    u2 = model.set_variable(var_type='_z', var_name='u2')
    u3 = model.set_variable(var_type='_z', var_name='u3')
    u4 = model.set_variable(var_type='_z', var_name='u4')
    
    model.set_rhs('u1', ct * omega_1**2)
    model.set_rhs('u2', ct * omega_2**2)
    model.set_rhs('u3', ct * omega_3**2)
    model.set_rhs('u4', ct * omega_4**2)

    # Combine the thrust forces into the input vector u
    T = model.set_variable(var_type='_z', var_name='T')
    u = vertcat(u1, u2, u3, u4)
    
    tau_x = model.set_variable(var_type='_z', var_name='tau_x')
    tau_y = model.set_variable(var_type='_z', var_name='tau_y')
    tau_z = model.set_variable(var_type='_z', var_name='tau_z')

    # Compute total thrust and torques
    T_tau = G @ u
    model.set_rhs('T', T_tau[0])
    model.set_rhs('tau_x', T_tau[1])
    model.set_rhs('tau_y', T_tau[2])
    model.set_rhs('tau_z', T_tau[3])

    # Define intertia's
    ixx = ctrl._getURDFParameter('ixx')
    iyy = ctrl._getURDFParameter('iyy')
    izz = ctrl._getURDFParameter('izz')

    I_v = diag(vertcat(ixx, iyy, izz))

    # TO-DO Waar komt het vandaan?
    Omega_cross = vertcat(
        omega_y * (izz * omega_z) - omega_z * (iyy * omega_y),
        omega_z * (ixx * omega_x) - omega_x * (izz * omega_z),
        omega_x * (iyy * omega_y) - omega_y * (ixx * omega_x)
    )
    omega_dot = inv(I_v) @ (vertcat(tau_x, tau_y, tau_z) - Omega_cross)
    model.set_rhs('omega_x', omega_dot[0])
    model.set_rhs('omega_y', omega_dot[1])
    model.set_rhs('omega_z', omega_dot[2])

    # Quaternions 
    Omega = vertcat(omega_x, omega_y, omega_z)  # Angular velocity vector

    M_q = vertcat(
        horzcat(-q_x, -q_y, -q_z),
        horzcat(q_w, -q_z, q_y),
        horzcat(q_z, q_w, -q_x),
        horzcat(-q_y, q_x, q_w)
    )
    q_dot = 0.5 * M_q @ Omega

    model.set_rhs('q_w', q_dot[0])
    model.set_rhs('q_x', q_dot[1])
    model.set_rhs('q_y', q_dot[2])
    model.set_rhs('q_z', q_dot[3])

    #### SYSTEM DYNAMICS HERE (x_k + 1) ####
    xi = vertcat(x, y, z)  # Concatenate state variables
    xi_dot = vertcat(x_dot, y_dot, z_dot)

    # Rotation matrix for quaternions
    R = vertcat(
        horzcat(1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)),
        horzcat(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_w * q_x)),
        horzcat(2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * (q_x**2 + q_y**2))
    )

    # Gravity and thrust
    thrust = (1/m) * T * (R @ vertcat(0, 0, 1))  # Thrust in world frame

    # Update dynamics
    xi_dot_new = xi_dot + (thrust + grav_vector) * dt

    # Set derivatives in the model
    model.set_rhs('x_dot', xi_dot_new[0])
    model.set_rhs('y_dot', xi_dot_new[1])
    model.set_rhs('z_dot', xi_dot_new[2])
    
    # Complete the model
    model.setup()

    # Configure the MPC
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
                'n_horizon': 10,
                't_step': 1/env.CTRL_FREQ,  # match with env control timestep
                'state_discretization': 'discrete',
                'store_full_solution': True
                }

    mpc.set_param(**setup_mpc) 

    # Set cost function
    model.set_expression(expr_name='cost', expr=cost)

    # Set contraints
    # lower bounds of the states
    mpc.bounds['lower','_x','x'] = -max_x
    # upper bounds of the states
    mpc.bounds['upper','_x','x'] = max_x

    mpc.setup()

    # Set initial conditions
    mpc.x0 = start_pos[0, :]
    mpc.u0 = np.zeros(3)
    mpc.set_initial_guess()

    #### Initialize Variables for Straight-Line Path ###########
    start_pos = start_pos
    target_pos = np.array([0, 0, 0]) #IMPLEMENT
    total_steps = int(duration_sec * control_freq_hz)
    step_size = (target_pos - start_pos) / total_steps  # Linear interpolation step

    #### Run the simulation 
    action = np.zeros((1, 4))
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        # Step the simulation 
        obs = env.step(action)
        action = mpc.make_step(obs)
        # Compute straight line target
        target_pos_step = start_pos + step_size * i

        target_pos = action[:3]  # Target position (x, y, z)
        target_vel = action[3:6]  # Target velocity (x_dot, y_dot, z_dot)

        # Compute Control Input 
        action, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs,           
            target_pos=target_pos_step,
            target_rpy=start_orient,  # Fixed orientation for simplicity
            target_vel=target_vel
        )

        # Log the Simulation 
        logger.log(
            drone=0,                      # Only one drone
            timestamp=i / env.CTRL_FREQ,
            state=obs,                 # Log the single drone state
            control=action          # Log the computed action
        )

        # Printout 
        env.render()

        # Sync the simulation 
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment 
    env.close()

    #### Save the simulation results 
    # Create a 'log' directory if it doesn't exist
    os.makedirs("log", exist_ok=True)

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"log/mpc_pid_static_{timestamp}.csv"

    # Save the CSV file
    logger.save_as_csv(filename)

    #### Plot the simulation results 
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
