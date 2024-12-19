"""Script demonstrating the joint use of simulation and acados mpc solver.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python3 acados_pybullet.py

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

from quadrotor_dynamic_model_test import exportModel
from acados_main import initialize_solver, set_initial_state, solve_ocp, get_solution

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
    x0 = np.concatenate([start_pos, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

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

    prediction_horizon = 20
    final_time = 10.0
    solver, nx, nu, prediction_horizon, final_time = initialize_solver(prediction_horizon=prediction_horizon, final_time=final_time, end_position = end_pos)

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
            set_initial_state(solver, state_vector)
            
            # Solve the OCP
            status = solve_ocp(solver)
            if status != 0:
                print(f"Acados solver failed at timestep {i}. State: {state_vector}")
                break

            simX, simU = get_solution(solver, nx, nu, prediction_horizon, final_time)

            predicted_x, predicted_y, predicted_z = simX[1, :3]
            predicted_x_dot, predicted_y_dot, predicted_z_dot = simX[1, 4:6]
            target_position = np.array([predicted_x, predicted_y, predicted_z]).flatten()
            target_velocity = np.array([predicted_x_dot, predicted_y_dot, predicted_z_dot]).flatten()

            # Write predictions to the file
            timestamp = i / env.CTRL_FREQ  # Time in seconds
            
            prediction_log_file.write(f"{timestamp},{predicted_x},{predicted_y},{predicted_z}\n")
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

            # action = action.reshape(1, 4)

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
