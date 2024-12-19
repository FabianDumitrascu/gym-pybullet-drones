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


def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = exportModel()
    ocp.model = model

    Tf = 1
    nx = model.x.rows()
    nu = model.u.rows()
    N = 20

    # set number of shooting intervals
    ocp.dims.N = N

    # set prediction horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    # cost matrices
    Q_mat = np.eye(13)
    R_mat = np.eye(4)

    # path cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    
    goal_position = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Adjust according to the model dimensions
    ocp.cost.yref = np.concatenate([goal_position, np.zeros(nu)])  # Concatenate goal position with zero controls (if needed)

    
    ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

    # terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = goal_position[:nx]  # Only the position part of the goal (adjust as needed)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q_mat

    # set constraints
    Fmax = 80
    ocp.constraints.lbu = np.array([-10, -10, -10 , -10])
    ocp.constraints.ubu = np.array([10, 10, 10, 10])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    print("idxbu:", ocp.constraints.idxbu)

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

    print(f"Final time (tf): {ocp.solver_options.tf}")
    print(f"Prediction horizon (N): {ocp.solver_options.N_horizon}")

    ocp.solver_options.print_level = 3  # Set higher print level for more diagnostics


    ocp_solver = AcadosOcpSolver(ocp)

    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        print(f"Solver failed with status: {status}")

    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")

    # Time array
    time = np.linspace(0, Tf, N+1)

    # Plot results
    plot_results(time, simX, simU)


if __name__ == '__main__':
    main()
