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
DEFAULT_CONTROL_FREQ_HZ = 12
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
    start_pos = np.array([0, 0, 1])
    end_pos = np.array([1, 0, 1])
    start_orient = np.array([0, 0, 0])
    INIT_RPYS = np.array([[0,0,0]])
    INIT_XYZS = np.array([start_pos])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
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
    model = do_mpc.model.Model('continuous')
    
    # Define constants
    grav_vector = np.array([0, 0, -9.8]).T
    m = 0.027 #kg
    ct = 3.16e-10
    cq = 7.94e-12
    l = 0.0397
    dt = 1/control_freq_hz
    ixx = 1.4e-5
    iyy = 1.4e-5
    izz = 2.17e-5
    I_v = diag(vertcat(ixx, iyy, izz))

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

    # Define G matrix
    G = np.array([[1, 1, 1, 1],
                  [l * np.sqrt(2)/2, -l * np.sqrt(2)/2, -l * np.sqrt(2)/2, l * np.sqrt(2)/2],
                  [-l * np.sqrt(2)/2, -l * np.sqrt(2)/2, l * np.sqrt(2)/2, l * np.sqrt(2)/2],
                  [cq/ct, -cq/ct, cq/ct, -cq/ct]])

    # Define rotor speeds (inputs)
    omega_1 = model.set_variable(var_type='_u', var_name='omega_1')
    omega_2 = model.set_variable(var_type='_u', var_name='omega_2')
    omega_3 = model.set_variable(var_type='_u', var_name='omega_3')
    omega_4 = model.set_variable(var_type='_u', var_name='omega_4')
    
    # Define thrust variables
    T = model.set_variable(var_type='_z', var_name='T')
    tau_x = model.set_variable(var_type='_z', var_name='tau_x')
    tau_y = model.set_variable(var_type='_z', var_name='tau_y')
    tau_z = model.set_variable(var_type='_z', var_name='tau_z')

    # Compute total thrust and torques
    u_vector = vertcat(omega_1**2, omega_2**2, omega_3**2, omega_4**2)  # 4x1 input vector
    G_u = G @ (ct * u_vector)  # Result is 4x1

    # Extract the scalar components explicitly
    model.set_alg('T', G_u[0])
    model.set_alg('tau_x', G_u[1])
    model.set_alg('tau_y', G_u[2])
    model.set_alg('tau_z', G_u[3])

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

    # Normalize the quaternions for numerical stability
    q_norm = sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2 + 1e-8)
    model.set_rhs('q_w', q_dot[0] / q_norm)
    model.set_rhs('q_x', q_dot[1] / q_norm)
    model.set_rhs('q_y', q_dot[2] / q_norm)
    model.set_rhs('q_z', q_dot[3] / q_norm)

    # Rotation matrix for quaternions
    R_matrix = vertcat(
        horzcat(1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)),
        horzcat(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_w * q_x)),
        horzcat(2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * (q_x**2 + q_y**2))
    )

    # Gravity and thrust
    thrust = (1/m) * T * (R_matrix @ vertcat(0, 0, 1))  # Thrust in world frame

    # Update dynamics
    xi_dot_dot = thrust + grav_vector

    # Set derivatives in the model
    model.set_rhs('x', x_dot)
    model.set_rhs('y', y_dot)
    model.set_rhs('z', z_dot)
    model.set_rhs('x_dot', xi_dot_dot[0])
    model.set_rhs('y_dot', xi_dot_dot[1])
    model.set_rhs('z_dot', xi_dot_dot[2])

    # u_prev = model.set_variable(var_type='_tvp', var_name='u_prev', shape=(4, 1))
    
    # Complete the model
    model.setup()

    # Configure the MPC
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
                'n_horizon': 10,
                't_step': 1/env.CTRL_FREQ,  # match with env control timestep
                'state_discretization': 'collocation',
                'store_full_solution': True
                }

    mpc.set_param(**setup_mpc) 

    mpc.set_param(t_step=1/env.CTRL_FREQ)

    # tvp_template = mpc.get_tvp_template()

    # # Time-varying parameter function
    # def tvp_fun(t_now):
    #     tvp_template['_tvp', :, 'u_prev'] = np.zeros((4, 1))  # Update with actual previous control input
    #     return tvp_template

    # mpc.set_tvp_fun(tvp_fun)

    # Set cost function
    Q_cost = diag([10.0, 10.0, 10.0])
    # R_cost = diag([0.1, 0.1, 0.1, 0.1]) 
    
    # lterm = (vertcat(x, y, z) - end_pos).T @ Q_cost @ (vertcat(x, y, z) - end_pos) + \
    #     ((vertcat(omega_1, omega_2, omega_3, omega_4) - u_prev) / dt).T @ R_cost @ ((vertcat(omega_1, omega_2, omega_3, omega_4) - u_prev) / dt)
    
    lterm = (vertcat(x, y, z) - end_pos.T).T @ Q_cost @ (vertcat(x, y, z) - end_pos.T)
    mterm = 0 * x 
    mpc.set_objective(lterm=lterm, mterm=mterm)

    # Set contraints
    # lower bounds of the states
    mpc.bounds['lower','_x','z'] = 0
    mpc.bounds['upper','_u','omega_1'] = 21000
    mpc.bounds['upper','_u','omega_2'] = 21000
    mpc.bounds['upper','_u','omega_3'] = 21000
    mpc.bounds['upper','_u','omega_4'] = 21000
    mpc.bounds['lower','_u','omega_1'] = 0
    mpc.bounds['lower','_u','omega_2'] = 0
    mpc.bounds['lower','_u','omega_3'] = 0
    mpc.bounds['lower','_u','omega_4'] = 0

    mpc.setup()

    # Set initial conditions
    x0 = np.array([
    0, 0, 1,         # Position (x, y, z)
    0, 0, 0,         # Velocity (x_dot, y_dot, z_dot)
    1, 0, 0, 0,      # Quaternion (q_w, q_x, q_y, q_z)
    0, 0, 0          # Angular velocity (omega_x, omega_y, omega_z)
    ])
    
    mpc.x0 = x0
    mpc.u0 = np.zeros(4)
    mpc.set_initial_guess()

    #### Initialize Variables for Straight-Line Path ###########
   # total_steps = int(duration_sec * control_freq_hz)

    #### Run the simulation 
    try:
        #### Run the simulation 
        action = np.zeros((1, 4))
        START = time.time()

        # Open a file to log predictions
        prediction_log_file = open("predictions_log.csv", "w")
        prediction_log_file.write("Time,Predicted_X,Predicted_Y,Predicted_Z\n")  # Header

        trajectory_log_file = open("full_trajectory_log.csv", "w")
        trajectory_log_file.write("Time,Step,X_Pred,Y_Pred,Z_Pred\n")  # Write header

        for i in range(0, int(duration_sec * env.CTRL_FREQ)):
            # Step the simulation 
            obs = env.step(action)[0]
            state_vector = (obs.flatten())[:13]
            mpc.make_step(state_vector)
            
            # Retrieve predicted states from do-mpc
            predicted_x = mpc.data.prediction(('_x', 'x'), t_ind=-1)[0, 0]
            predicted_y = mpc.data.prediction(('_x', 'y'), t_ind=-1)[0, 0]
            predicted_z = mpc.data.prediction(('_x', 'z'), t_ind=-1)[0, 0]

            predicted_x_dot = mpc.data.prediction(('_x', 'x_dot'), t_ind=-1)[0, 0]
            predicted_y_dot = mpc.data.prediction(('_x', 'y_dot'), t_ind=-1)[0, 0]
            predicted_z_dot = mpc.data.prediction(('_x', 'z_dot'), t_ind=-1)[0, 0]

            predicted_x_traj = mpc.data.prediction(('_x', 'x'))[:, 0]  # All future x predictions
            predicted_y_traj = mpc.data.prediction(('_x', 'y'))[:, 0]  # All future y predictions
            predicted_z_traj = mpc.data.prediction(('_x', 'z'))[:, 0]  # All future z predictions

            target_position = np.array([predicted_x, predicted_y, predicted_z]).flatten()
            target_velocity = np.array([predicted_x_dot, predicted_y_dot, predicted_z_dot]).flatten()

            # Write predictions to the file
            timestamp = i / env.CTRL_FREQ  # Time in seconds
            prediction_log_file.write(f"{timestamp},{predicted_x},{predicted_y},{predicted_z}\n")
            for step, (px, py, pz) in enumerate(zip(predicted_x_traj, predicted_y_traj, predicted_z_traj)):
                trajectory_log_file.write(f"{timestamp},{step},{px},{py},{pz}\n")

            print("Target Position:", target_position)
            print("Target Velocity:", target_velocity)

            # Compute Control Input 
            action, _, _ = ctrl.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=state_vector,           
                target_pos=target_position,
                target_rpy=start_orient,  # Fixed orientation
                target_vel=target_velocity
            )

            action = action.reshape(1, 4)

            # Pad control input to size (12,)
            control_padded = np.zeros(12)  # Create a 12-element array
            control_padded[:4] = action.flatten()  # Place rotor speeds in the first 4 elements

            # Log the Simulation 
            logger.log(
                drone=0,                      # Only one drone
                timestamp=i / env.CTRL_FREQ,
                state=obs.flatten(),          # Log the single drone state
                control=control_padded        # Log the computed action
            )

            # Render
            env.render()

            # Sync the simulation 
            if gui:
                sync(i, START, env.CTRL_TIMESTEP)

    except KeyboardInterrupt:
        print("Simulation interrupted. Saving logs...")

    finally:
        #### Close the environment and save logs
        env.close()

        os.makedirs("log", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"log/mpc_pid_static_{timestamp}.csv"
        logger.save_as_csv(filename)
        print(f"Log saved to {filename}")

        # Close the file
        prediction_log_file.close()
        print("Predictions log saved to 'predictions_log.csv'")
        trajectory_log_file.close()

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
