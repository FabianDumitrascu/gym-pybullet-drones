from acados_template import AcadosOcp, AcadosOcpSolver
from quadrotor_dynamic_model_test import exportModel
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import time


def plot_results(time, simX, simU):
    """
    Plot all state and control trajectories in separate combined plots.
    """
    nx = simX.shape[1]
    nu = simU.shape[1]

    # Plot all states in one figure
    plt.figure(figsize=(12, 6))
    for i in range(nx):
        plt.plot(time, simX[:, i], label=f"x[{i}]")
    plt.xlabel("Time (s)")
    plt.ylabel("States")
    plt.grid()
    plt.legend()
    plt.title("State Trajectories")
    plt.show()

    # Plot all control inputs in one figure
    plt.figure(figsize=(12, 6))
    for i in range(nu):
        plt.step(time[:-1], simU[:, i], label=f"u[{i}]")
    plt.xlabel("Time (s)")
    plt.ylabel("Controls")
    plt.grid()
    plt.legend()
    plt.title("Control Inputs")
    plt.show()

def initialize_solver(prediction_horizon=20, final_time=1.0, end_position=np.zeros(3), x0=np.zeros(13)):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = exportModel()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    end_state = np.concatenate([end_position, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # set dimensions and prediction horizon
    ocp.dims.N = prediction_horizon
    ocp.solver_options.N_horizon = prediction_horizon
    ocp.solver_options.tf = final_time
    ocp.solver_options.nlp_solver_max_iter = 200 

    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.nlp_solver_tol_eq = 1e-6
    ocp.solver_options.nlp_solver_tol_ineq = 1e-6
    ocp.solver_options.nlp_solver_tol_comp = 1e-6

    # cost matrices
    Q_mat = np.diag([
                    5, 5, 5,    # x, y, z
                    1, 1, 10,       # vx, vy, vz
                    0.1, 0.1, 0.1, 0.1,  # q0, q1, q2, q3 (orientation)
                    1, 1, 1  # wx, wy, wz
    ])
    R_mat = 10*np.eye(4)

    hover_thrust = 0.06615
    end_u = np.array([hover_thrust, hover_thrust, hover_thrust, hover_thrust])
    # path cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.cost.yref = np.concatenate([end_state, end_u])
    ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

    # terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = end_state[:nx]
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q_mat
    
    ocp.constraints.idxbx_e = np.arange(nx)  # Constrain all states at terminal time
    end_tolerance = 0.05  # Define tolerance for terminal constraints
    ocp.constraints.lbx_e = end_state - end_tolerance  # Lower bounds
    ocp.constraints.ubx_e = end_state + end_tolerance  # Upper bounds

    # set constraints
    ocp.constraints.lbu = np.array([0, 0, 0, 0])
    max_thrust = 0.15  
    ocp.constraints.ubu = np.array([max_thrust, max_thrust, max_thrust, max_thrust])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.idxbx_0 = np.arange(nx)    
    ocp.constraints.lbx_0 = x0            
    ocp.constraints.ubx_0 = x0

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

    ocp.solver_options.print_level = 0 # Set higher print level for more diagnostics

    solver = AcadosOcpSolver(ocp)

    return solver, nx, nu, prediction_horizon, final_time

def set_initial_state(solver, state_vector, input_vector, prediction_horizon):
    solver.set(0, "x", state_vector)
    solver.set(0, "lbx", state_vector)
    solver.set(0, "ubx", state_vector)
    # solver.set(0, "u", input_vector)
    # solver.set(0, "lbu", input_vector - 0.2*input_vector)
    # solver.set(0, "ubu", input_vector + 0.2*input_vector)
    # print('acados state = ', state_vector[:3])

    # for stage in range(prediction_horizon):
    #     solver.set(stage, "u", input_vector)
    retrieved_state = solver.get(0, "x")
    print(f"acados state = {retrieved_state[0:3]}")

def solve_ocp(solver, simX_prev=None, simU_prev=None, prediction_horizon=None):
    # Set the previous solution as the initial guess for warm-starting
    if simX_prev is not None and simU_prev is not None:
        for i in range(prediction_horizon):
            solver.set(i, "x", simX_prev[i, :])  # Warm-start states
            solver.set(i, "u", simU_prev[i, :])  # Warm-start controls
        solver.set(prediction_horizon, "x", simX_prev[prediction_horizon, :])  # Final state

    # Solve the optimization problem
    status = solver.solve()
    if status != 0:
        print(f"Solver failed with status: {status}")
    return status

def get_solution(solver, nx, nu, prediction_horizon, final_time):
    simX = np.zeros((prediction_horizon + 1, nx))
    simU = np.zeros((prediction_horizon, nu))

    # get solution
    for i in range(prediction_horizon):
        simX[i,:] = solver.get(i, "x")
        simU[i,:] = solver.get(i, "u")
    simX[prediction_horizon,:] = solver.get(prediction_horizon, "x")

    # Plot results
    time = np.linspace(0, final_time, prediction_horizon+1)
    plot_results(time, simX[:,0:3], simU)

    return simX, simU

def add_debug_dot():
    """
    Add a dot in the PyBullet simulation to represent the drone's position.
    """
    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.8)

    # Load your drone model
    drone = p.loadURDF("quadrotor.urdf", [0, 0, 1])

    # Simulate
    for i in range(1000):
        # Step simulation
        p.stepSimulation()
        time.sleep(1 / 240.0)

        # Get the drone's current position
        pos, _ = p.getBasePositionAndOrientation(drone)

        # Draw a small point (it will update each frame)
        p.addUserDebugLine(pos, pos, [1, 0, 0], 5)  # Red dot (size 5)

    p.disconnect()

# if __name__ == '__main__':
#     main()
