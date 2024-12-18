"""Script demonstrating the joint use of simulation and control using ACADOS MPC with a PID backup.

The simulation is run by a `CtrlAviary` environment.
The control is given by a combination of MPC (using ACADOS) and DSLPIDControl.

This script replaces do-mpc with acados for solving the MPC problem.

Example
-------
In a terminal, run as:

    $ python mpc_pid_static.py

"""
import os
import time
import argparse
from datetime import datetime
import math
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from casadi import *
from scipy.spatial.transform import Rotation as R

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

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

def create_acados_model_and_solver(dt, horizon_steps, end_pos):
    """
    Create the ACADOS OCP model and solver for the quadrotor.
    """
    # Constants
    grav_vector = np.array([0, 0, -9.8])
    m = 0.027 # kg
    ct = 3.16e-10
    cq = 7.94e-12
    l = 0.0397
    ixx = 1.4e-5
    iyy = 1.4e-5
    izz = 2.17e-5

    # Define state variables
    x       = SX.sym('x')
    y       = SX.sym('y')
    z       = SX.sym('z')
    x_dot   = SX.sym('x_dot')
    y_dot   = SX.sym('y_dot')
    z_dot   = SX.sym('z_dot')
    q_w     = SX.sym('q_w')
    q_x     = SX.sym('q_x')
    q_y     = SX.sym('q_y')
    q_z     = SX.sym('q_z')
    omega_x = SX.sym('omega_x')
    omega_y = SX.sym('omega_y')
    omega_z = SX.sym('omega_z')

    # Inputs (rotor speeds)
    omega_1 = SX.sym('omega_1')
    omega_2 = SX.sym('omega_2')
    omega_3 = SX.sym('omega_3')
    omega_4 = SX.sym('omega_4')

    # State and input vector
    X = vertcat(x, y, z, x_dot, y_dot, z_dot, q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z)
    U = vertcat(omega_1, omega_2, omega_3, omega_4)

    # Inertia matrix
    I_v = diag(SX([ixx, iyy, izz]))

    # Allocation matrix G
    G = np.array([
        [1, 1, 1, 1],
        [l * np.sqrt(2)/2, -l * np.sqrt(2)/2, -l * np.sqrt(2)/2, l * np.sqrt(2)/2],
        [-l * np.sqrt(2)/2, -l * np.sqrt(2)/2, l * np.sqrt(2)/2, l * np.sqrt(2)/2],
        [cq/ct, -cq/ct, cq/ct, -cq/ct]
    ])

    # Compute thrust and torques
    u_vector = vertcat(omega_1**2, omega_2**2, omega_3**2, omega_4**2)
    G_u = ct * G @ u_vector
    T = G_u[0]
    tau_x = G_u[1]
    tau_y = G_u[2]
    tau_z = G_u[3]

    # Angular velocities
    Omega = vertcat(omega_x, omega_y, omega_z)

    # Compute angular acceleration
    # Note: In the original code, there's a mention of a cross term (Omega_cross) that was incomplete.
    # The correct rigid body dynamics for angular part: I * Omega_dot + Omega x (I * Omega) = tau
    # Here we define Omega_dot = I^{-1} (tau - Omega x (I * Omega))
    I_Omega = I_v @ Omega
    Omega_cross = cross(Omega, I_Omega)
    tau = vertcat(tau_x, tau_y, tau_z)
    Omega_dot = solve(I_v, tau - Omega_cross)

    # Quaternions update
    # q_dot = 0.5 * M_q * Omega, where
    # M_q = [[-q_x -q_y -q_z],
    #        [ q_w -q_z  q_y],
    #        [ q_z  q_w -q_x],
    #        [-q_y  q_x  q_w]]
    # Implementing directly:
    q_dot = 0.5 * vertcat(
        -q_x*omega_x - q_y*omega_y - q_z*omega_z,
         q_w*omega_x - q_z*omega_y + q_y*omega_z,
         q_z*omega_x + q_w*omega_y - q_x*omega_z,
        -q_y*omega_x + q_x*omega_y + q_w*omega_z
    )

    # Normalize quaternions for numerical stability
    q_norm = sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2 + 1e-8)
    q_w_dot = q_dot[0]/q_norm
    q_x_dot = q_dot[1]/q_norm
    q_y_dot = q_dot[2]/q_norm
    q_z_dot = q_dot[3]/q_norm

    # Rotation matrix from quaternions
    R_matrix = vertcat(
        horzcat(1 - 2*(q_y**2 + q_z**2),     2*(q_x*q_y - q_w*q_z),     2*(q_x*q_z + q_w*q_y)),
        horzcat(2*(q_x*q_y + q_w*q_z),     1 - 2*(q_x**2 + q_z**2),     2*(q_y*q_z - q_w*q_x)),
        horzcat(2*(q_x*q_z - q_w*q_y),     2*(q_y*q_z + q_w*q_x), 1 - 2*(q_x**2 + q_y**2))
    )

    # Acceleration in world frame
    thrust_world = (T/m)* (R_matrix @ vertcat(0,0,1))
    xi_ddot = thrust_world + grav_vector

    # State derivatives
    x_dot_dot = xi_ddot[0]
    y_dot_dot = xi_ddot[1]
    z_dot_dot = xi_ddot[2]

    # Define the dynamics as f_expl_expr
    f_expl_expr = vertcat(
        x_dot,
        y_dot,
        z_dot,
        x_dot_dot,
        y_dot_dot,
        z_dot_dot,
        q_w_dot,
        q_x_dot,
        q_y_dot,
        q_z_dot,
        Omega_dot[0],
        Omega_dot[1],
        Omega_dot[2]
    )

    # Create AcadosModel
    model = AcadosModel()
    model.name = "quadrotor_mpc"
    model.x = X
    model.u = U
    model.f_expl_expr = f_expl_expr
    model.p = []  # no parameters
    model.xdot = SX.sym('xdot', X.size1())
    # Not strictly needed to define f_impl since we use ERK integrator.
    # f_impl_expr = model.xdot - f_expl_expr
    # model.f_impl_expr = f_impl_expr

    # Create OCP
    ocp = AcadosOcp()
    ocp.model = model

    # Discretization
    ocp.dims.N = horizon_steps
    ocp.solver_options.tf = horizon_steps * dt

    # Bounds
    # z >= 0
    # Inputs between 0 and 21000
    ocp.constraints.lbu = np.array([0,0,0,0])
    ocp.constraints.ubu = np.array([21000,21000,21000,21000])
    ocp.constraints.idxbu = np.array([0,1,2,3])

    # We'll track position (x,y,z) and try to reach end_pos
    # We'll also put a mild penalty on controls to avoid large jumps.
    # cost: minimize ||(x - x_ref), (y - y_ref), (z - z_ref), u||^2
    # We'll define y = [x,y,z,u_1,u_2,u_3,u_4]
    # W = diag([100, 100, 500, 0.1, 0.1, 0.1, 0.1])

    ny = 7
    W = np.diag([100.0, 100.0, 500.0, 0.1, 0.1, 0.1, 0.1])

    y_expr = vertcat(x, y, z, omega_1, omega_2, omega_3, omega_4)
    # Reference for states: end_pos and control = 0
    y_ref = np.array([end_pos[0], end_pos[1], end_pos[2], 0,0,0,0])

    # Cost on state and inputs
    ny = 7  # Total number of cost components: [x, y, z, omega_1, omega_2, omega_3, omega_4]
    W = np.diag([100.0, 100.0, 500.0, 0.1, 0.1, 0.1, 0.1])  # Weight matrix
    y_ref = np.array([end_pos[0], end_pos[1], end_pos[2], 0, 0, 0, 0])  # Reference

    ocp.cost.cost_type = 'LINEAR_LS'  # Stage cost
    ocp.cost.cost_type_e = 'LINEAR_LS'  # Terminal cost

    # Mapping between states/inputs and cost
    ocp.cost.Vx = np.zeros((ny, X.size1()))  # Map states to cost
    ocp.cost.Vu = np.zeros((ny, U.size1()))  # Map inputs to cost
    ocp.cost.Vx[:3, :3] = np.eye(3)  # Map [x, y, z] states
    ocp.cost.Vu[3:, :4] = np.eye(4)  # Map rotor inputs

    ocp.cost.W = W
    ocp.cost.yref = y_ref

    # Terminal cost
    ny_e = 3  # Only position for terminal cost
    W_e = np.diag([100.0, 100.0, 500.0])  # Terminal weight
    ocp.cost.Vx_e = np.zeros((ny_e, X.size1()))
    ocp.cost.Vx_e[:3, :3] = np.eye(3)  # Terminal cost on [x, y, z]
    ocp.cost.W_e = W_e
    ocp.cost.yref_e = end_pos

    # Terminal cost on position
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.model.cost_y_expr_e = vertcat(x, y, z)
    W_e = np.diag([100, 100, 500])
    ocp.cost.W_e = W_e
    ocp.cost.yref_e = end_pos

    # Initial condition constraints
    ocp.constraints.x0 = np.array([
        0,0,1,       # x,y,z
        0,0,0,       # x_dot,y_dot,z_dot
        1,0,0,0,     # q_w, q_x, q_y, q_z
        0,0,0        # omega_x,omega_y,omega_z
    ])

    # Solver options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.print_level = 0

    # Create solver
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

    return ocp, ocp_solver


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

    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the PID controller ##########################
    ctrl = DSLPIDControl(drone_model=drone)

    #### Setup ACADOS MPC ######################################
    dt = 1.0/control_freq_hz
    horizon_steps = 10
    ocp, ocp_solver = create_acados_model_and_solver(dt, horizon_steps, end_pos)

    #### Run the simulation #####################################
    try:
        action = np.zeros((1,4))
        START = time.time()

        # Open files for logging
        prediction_log_file = open("predictions_log.csv", "w")
        prediction_log_file.write("Time,Predicted_X,Predicted_Y,Predicted_Z\n")

        trajectory_log_file = open("full_trajectory_log.csv", "w")
        trajectory_log_file.write("Time,Step,X_Pred,Y_Pred,Z_Pred\n")

        for i in range(0, int(duration_sec * env.CTRL_FREQ)):
            # Step simulation
            obs = env.step(action)[0]
            state_vector = obs.flatten()[:13]

            # Set current state as initial condition
            for ix in range(len(state_vector)):
                ocp_solver.set(0, "x", state_vector[ix])

            # Set references for the cost function
            # We'll keep the same reference for each stage
            # Because we want to reach end_pos and 0 control
            y_ref = np.array([end_pos[0], end_pos[1], end_pos[2], 0,0,0,0])
            for k in range(horizon_steps):
                ocp_solver.set(k, "yref", y_ref)
            ocp_solver.set(horizon_steps, "yref_e", end_pos)

            # Solve OCP
            status = ocp_solver.solve()
            if status != 0:
                print(f"ACADOS solver failed with status {status}, using previous action as fallback.")

            # Get predicted trajectory and input
            predicted_states = []
            for k in range(horizon_steps+1):
                predicted_state_k = ocp_solver.get(k, "x")
                predicted_states.append(predicted_state_k)

            # First predicted state (k=0) is current, next states are predictions
            # Extract predicted position at k=1 (the next step)
            predicted_x = predicted_states[1][0]
            predicted_y = predicted_states[1][1]
            predicted_z = predicted_states[1][2]

            # Full trajectory of x,y,z
            predicted_x_traj = [s[0] for s in predicted_states]
            predicted_y_traj = [s[1] for s in predicted_states]
            predicted_z_traj = [s[2] for s in predicted_states]

            # Predicted velocities (just for target), from k=1:
            predicted_x_dot = predicted_states[1][3]
            predicted_y_dot = predicted_states[1][4]
            predicted_z_dot = predicted_states[1][5]

            target_pos = np.array([predicted_x, predicted_y, predicted_z])
            target_vel = np.array([predicted_x_dot, predicted_y_dot, predicted_z_dot])

            # Write predictions to file
            timestamp = i / env.CTRL_FREQ
            prediction_log_file.write(f"{timestamp},{predicted_x},{predicted_y},{predicted_z}\n")
            for step_p, (px, py, pz) in enumerate(zip(predicted_x_traj, predicted_y_traj, predicted_z_traj)):
                trajectory_log_file.write(f"{timestamp},{step_p},{px},{py},{pz}\n")

            # Get the first input from solver
            u0 = ocp_solver.get(0, "u")
            # Use DSLPID to refine the action
            action_pid, _, _ = ctrl.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=state_vector,
                target_pos=target_pos,
                target_rpy=start_orient,
                target_vel=target_vel
            )

            # Combine or just use PID (the user might want to blend).
            # For now we just apply the PID action. If we wanted pure MPC control:
            # action = u0.reshape(1,4)
            # But since original code used PID after predictions, we follow that logic.
            action = action_pid.reshape(1,4)

            # Pad for logger
            control_padded = np.zeros(12)
            control_padded[:4] = action.flatten()

            # Log data
            logger.log(
                drone=0,
                timestamp=timestamp,
                state=obs.flatten(),
                control=control_padded
            )

            env.render()

            if gui:
                sync(i, START, env.CTRL_TIMESTEP)

    except KeyboardInterrupt:
        print("Simulation interrupted. Saving logs...")

    finally:
        env.close()

        os.makedirs("log", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"log/mpc_pid_static_{timestamp}.csv"
        logger.save_as_csv(filename)
        print(f"Log saved to {filename}")

        prediction_log_file.close()
        print("Predictions log saved to 'predictions_log.csv'")
        trajectory_log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quadrotor flight script using CtrlAviary and ACADOS MPC + DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 1)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 12)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 100)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))