from acados_template import AcadosOcp, AcadosOcpSolver
from quadrotor_dynamic_model_test import exportModel
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

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

def initialize_solver(prediction_horizon=20, final_time=1.0, end_position=np.zeros(0)):
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

    # cost matrices
    Q_mat = 10*np.eye(13)
    R_mat = np.eye(4)

    # path cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.cost.yref = np.concatenate([end_state, np.zeros(nu)])
    ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

    # terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = end_state[:nx]
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q_mat

    # set constraints
    ocp.constraints.lbu = np.array([0, 0, 0, 0])
    ocp.constraints.ubu = np.array([1000, 1000, 1000, 1000])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # placeholder x0

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

    # print(f"Final time (tf): {ocp.solver_options.tf}")
    # print(f"Prediction horizon (N): {ocp.solver_options.N_horizon}")

    ocp.solver_options.print_level = 0 # Set higher print level for more diagnostics

    solver = AcadosOcpSolver(ocp)

    return solver, nx, nu, prediction_horizon, final_time

def set_initial_state(solver, x0):
    solver.set(0, "x", x0)

def solve_ocp(solver):
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
    # plot_results(time, simX, simU)

    return simX, simU


# if __name__ == '__main__':
#     main()
