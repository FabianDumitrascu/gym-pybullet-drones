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
import time
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
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
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 100
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# Define start and end postion
start_pos = np.array([0,0,0.5])
end_pos = np.array([0.5,0.5,1])

def target_trajectory_generator(start_pos, end_pos):
    distance = np.linalg.norm(start_pos - end_pos)
    dist_points = 0.5
    num_points = int(distance / dist_points)
    waypoints = np.empty((0, 3))
    for i in range(num_points+1):
        x = start_pos[0] + (end_pos[0]-start_pos[0]) / num_points * i
        y = start_pos[1] + (end_pos[1]-start_pos[1]) / num_points * i
        z = start_pos[2] + (end_pos[2]-start_pos[2]) / num_points * i
        waypoint = np.array([[x, y, z]])
        waypoints = np.vstack((waypoints, waypoint))
    return waypoints

def thrust_to_rpm(thrusts):
    # Parameters
    kf = 3.16e-10 # Thrust coefficient from URDF
    PWM2RPM_SCALE = 0.2685
    PWM2RPM_CONST = 4070.3
    MIN_PWM = 20000  # Minimum PWM (µs)
    MAX_PWM = 65535  # Maximum PWM (µs)

    # Convert thrust to RPM
    rpm = np.sqrt(thrusts / kf)  # Calculate angular velocity (rad/s)

    print(f"Thrusts = {thrusts[0,:]}")
    print(f"RPMs = {rpm[0,:]}")

    return rpm

def plot_multi_time(data_matrix, sim_time, labels=None, plot_title=None, second_data_matrix=None, primary_label="Primary", secondary_label="Secondary"):
    """
    Parameters:
    - data_matrix: np.ndarray
        A 2D numpy array where each row represents a time step, and each column represents a different value.
    - sim_time: float
        Total simulation duration in seconds.
    - labels: list of str (optional)
        Labels corresponding to each value being plotted. Must match the number of columns in the data_matrix.
    - plot_title: str (optional)
        Title for the plot.
    - second_data_matrix: np.ndarray (optional)
        A second 2D numpy array for comparison, with the same shape as data_matrix.
    - primary_label: str
        Label for the primary data.
    - secondary_label: str
        Label for the secondary data.
    """
    # Time array
    num_points = data_matrix.shape[0]
    time = np.linspace(0, sim_time, num_points)

    num_values = data_matrix.shape[1]

    fig, axes = plt.subplots(num_values, 1, figsize=(10, 5 * num_values), sharex=True)
    if num_values == 1:
        axes = [axes]

    for i in range(num_values):
        values = data_matrix[:, i]
        axes[i].plot(time, values, label=primary_label, color="blue")
        if second_data_matrix:
            second_values = second_data_matrix[:, i]
            axes[i].plot(time, second_values, label=secondary_label, color="orange", linestyle="--")

        # Label axes
        label = labels[i] if labels and i < len(labels) else f"Val {i + 1}"
        axes[i].set_ylabel(label)
        axes[i].grid(True)
        if second_data_matrix:
            axes[i].legend()

    axes[-1].set_xlabel("Time (s)")

    if plot_title:
        fig.suptitle(plot_title, fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def add_custom_obstacles(client):
    '''Add custom obstacles and retrieve their boundary data.'''
    # Load a sphere
    sphere_id = p.loadURDF("../assets/sphere.urdf", [0.25, 0.25, 0.7], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=client)

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
        colab=DEFAULT_COLAB,
        start_pos=start_pos,
        end_pos=end_pos
        ):
    #### Initialize the simulation #############################

    # Define spherical obstacle
    sphere_radius = 0.15
    sphere_center = np.array([0.25, 0.25, 0.7])

    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])
    INIT_XYZS = np.array([start_pos])
    x0 = np.concatenate([start_pos, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

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
    
    # Add custom obstacles
    add_custom_obstacles(env.getPyBulletClient())

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    ctrl = DSLPIDControl(drone_model=drone)
    prediction_horizon = 20
    final_time = 6

    solver, nx, nu, prediction_horizon, final_time = initialize_solver(prediction_horizon=prediction_horizon, final_time=final_time, end_position = end_pos, x0=x0, sphere_radius=sphere_radius, sphere_center=sphere_center)
    
    hover_rpm = 14468.43
    hover_u = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    simX_prev = np.tile(x0, (prediction_horizon + 1, 1))
    simU_prev = np.tile(hover_u, (prediction_horizon, 1))
    
    set_initial_state(solver, x0, hover_u, prediction_horizon)

    #### Run the simulation 
    try: 
        # Start at hover speed
        action = hover_u.reshape(1, 4)
        
        #### Run the simulation
        START = time.time()

        target_position = start_pos
        waypoint_index = 1

        for i in range(0, int(duration_sec * env.CTRL_FREQ)):
            # Step the simulation 
            obs = env.step(action)[0]   
            print(f"Current rpms for : {env.current_rpms[0]}")
            state_vector = (obs.flatten())[:13]
        
            # set_initial_state(solver, state_vector, hover_u, prediction_horizon)

            if i == 0 :
                set_initial_state(solver, state_vector, hover_u, prediction_horizon)

                
                # Solve the OCP
                status = solve_ocp(solver, simX_prev, simU_prev, prediction_horizon)
                if status not in [0, 2]:   # 0 = success, 2 = max iters but let's accept
                    print(f"ACADOS gave an unexpected status: {status}, stopping.")
                    break
                simX, simU = get_solution(solver, nx, nu, prediction_horizon, final_time, sphere_radius, sphere_center, start_pos, end_pos)
                print('simU[0] = ', simU[0,:])

            predicted_x, predicted_y, predicted_z = simX[waypoint_index, :3]
            target_position = np.array([predicted_x, predicted_y, predicted_z]).flatten()

            # plot_multi_time(simX[:,:5], 1)

            # Add debug dot for predicted position
            p.addUserDebugLine(
                lineFromXYZ=target_position,
                lineToXYZ=target_position + np.array([0, 0, 0.1]),
                lineColorRGB=[1, 0, 0],  # Red color
                lineWidth=10,
                lifeTime=1/env.CTRL_FREQ
            )

            print('timestep = ', i)
            # print('observation = ', obs)
            print('state_vector = ', state_vector[:3])
            print("Target Position:", target_position)

            # Compute Control Input 
            action, _, _ = ctrl.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=state_vector,           
                target_pos=target_position,
            )

            drone_position = state_vector[:3]

            p.addUserDebugLine(
                lineFromXYZ=drone_position,
                lineToXYZ=drone_position + np.array([0, 0, 0.1]),
                lineColorRGB=[0, 0, 1],  # Blue color
                lineWidth=3,
                lifeTime=1/env.CTRL_FREQ
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

            if np.linalg.norm(state_vector[:3] - target_position) < 0.05 and waypoint_index < prediction_horizon:
                waypoint_index += 1

    except KeyboardInterrupt:
        print("Simulation interrupted. Saving logs...")

    finally:
        #### Close the environment and save logs
        env.close()

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


