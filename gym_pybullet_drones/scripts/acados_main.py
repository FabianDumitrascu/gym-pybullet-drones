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

def plot_results_2d_3d(time, simX, simU, sphere_radius, sphere_center, start_pos, end_pos):
    """
    Plot all state and control trajectories in separate combined plots,
    then do an additional x-y top-down plot with the obstacle region.
    """
    nx = simX.shape[1]
    nu = simU.shape[1]

    # Top down XY plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(simX[:, 0], simX[:, 1], 'b.-', label="(x,y) path")
    # Plot straight-line trajectory
    ax.plot(
        [start_pos[0], end_pos[0]],
        [start_pos[1], end_pos[1]],
        'g--',
        label="Straight Trajectory"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("XY Top view Path plus obstacle region")
    ax.grid(True)
    ax.axis('equal')

    # Draw the obstacle region (circle).
    from matplotlib.patches import Circle
    obstacle_circle = Circle((sphere_center[0], sphere_center[1]), sphere_radius, color="red", alpha=0.3, label="Obstacle Projection")
    ax.add_patch(obstacle_circle)
    ax.legend()
    plt.show()

    # Side view XZ
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(simX[:, 0], simX[:, 2], 'b.-', label="(x,z) path")
    # Plot straight-line trajectory
    ax.plot(
        [start_pos[0], end_pos[0]],
        [start_pos[2], end_pos[2]],
        'g--',
        label="Straight Trajectory"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("XZ Side view Path plus obstacle region")
    ax.grid(True)
    ax.axis('equal')

    # Draw the obstacle region (circle).
    from matplotlib.patches import Circle
    obstacle_circle = Circle((sphere_center[1], sphere_center[2]), sphere_radius, color="red", alpha=0.3, label="Obstacle Projection")
    ax.add_patch(obstacle_circle)
    ax.legend()
    plt.show()

    # Side view YZ
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(simX[:, 0], simX[:, 2], 'b.-', label="(y,z) path")

    # Plot straight-line trajectory
    ax.plot(
        [start_pos[1], end_pos[1]],
        [start_pos[2], end_pos[2]],
        'g--',
        label="Straight Trajectory"
    )

    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("YZ Side view Path plus obstacle region")
    ax.grid(True)
    ax.axis('equal')

    # Draw the obstacle region (circle).
    from matplotlib.patches import Circle
    obstacle_circle = Circle((sphere_center[0], sphere_center[1]), sphere_radius, color="red", alpha=0.3, label="Obstacle Projection")
    ax.add_patch(obstacle_circle)
    ax.legend()
    plt.show()

    # 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(simX[:, 0], simX[:, 1], simX[:, 2], 'b.-', label="Drone Path")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D Path with Obstacle Sphere")
    ax.grid(True)

    # Draw the sphere in 3D
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = sphere_radius * np.outer(np.cos(u), np.sin(v)) + sphere_center[0]
    y = sphere_radius * np.outer(np.sin(u), np.sin(v)) + sphere_center[1]
    z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + sphere_center[2]
    ax.plot_surface(x, y, z, color='red', alpha=0.3, label="Obstacle Sphere")

    ax.legend()
    plt.show()

def initialize_solver(prediction_horizon=20, final_time=1.0, end_position=np.zeros(3), x0=np.zeros(13), sphere_radius=0.1, sphere_center= np.array([0,0,1])):
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
    ocp.solver_options.nlp_solver_max_iter = 600 

    ocp.solver_options.nlp_solver_tol_stat = 1e-3
    ocp.solver_options.nlp_solver_tol_eq = 1e-3
    ocp.solver_options.nlp_solver_tol_ineq = 1e-3
    ocp.solver_options.nlp_solver_tol_comp = 1e-3

    # cost matrices
    Q_mat = np.diag([
                    5, 5, 10,    # x, y, z
                    1, 1, 5,       # vx, vy, vz
                    1, 1, 1, 1,  # q0, q1, q2, q3 (orientation)
                    1, 1, 1  # wx, wy, wz
    ])
    R_mat = 30*np.eye(4)

    hover_thrust = 0.06615
    end_u = np.array([hover_thrust, hover_thrust, hover_thrust, hover_thrust])
    # path cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.cost.yref = np.concatenate([end_state, end_u])
    ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

    # terminal cost
    Q_mat_e = np.diag([
                    2, 2, 2,    # x, y, z
                    1, 1, 10,       # vx, vy, vz
                    0, 0, 0, 0,  # q0, q1, q2, q3 (orientation)
                    1, 1, 1  # wx, wy, wz
    ])

    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = end_state[:nx]
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q_mat_e
    
    # Add obstacles
    idxbx_e = np.array([0, 1, 2, 3, 4, 5])  # x, y, z, vx, vy, vz
    ocp.constraints.idxbx_e = idxbx_e

    end_tolerance = 0.05
    lbx_e = end_state[idxbx_e] - end_tolerance
    ubx_e = end_state[idxbx_e] + end_tolerance
    ocp.constraints.lbx_e = lbx_e
    ocp.constraints.ubx_e = ubx_e

    # Create symbolic expressions for multiple constraints
    x = ocp.model.x[0]  # x position
    y = ocp.model.x[1]  # y position
    z = ocp.model.x[2]  # z position

    dist_expr = ca.sqrt((x - sphere_center[0])**2 + (y -sphere_center[1])**2 + (z - sphere_center[2])**2)
    ocp.model.con_h_expr = sphere_radius - dist_expr  # <= 0
    ocp.dims.nh = 1
    ocp.constraints.lh = np.array([-1e6])
    ocp.constraints.uh = np.array([0.0])

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

    ocp.solver_options.print_level = 1 # Set higher print level for more diagnostics

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
    if status not in [0, 2]:   # 0 = success, 2 = max iters but let's accept
        print(f"ACADOS gave an unexpected status: {status}, stopping.")

    return status

def get_solution(solver, nx, nu, prediction_horizon, final_time, sphere_radius, sphere_center, start_pos, end_pos):
    simX = np.zeros((prediction_horizon + 1, nx))
    simU = np.zeros((prediction_horizon, nu))

    # get solution
    for i in range(prediction_horizon):
        simX[i,:] = solver.get(i, "x")
        simU[i,:] = solver.get(i, "u")
    simX[prediction_horizon,:] = solver.get(prediction_horizon, "x")

    # Plot results
    time = np.linspace(0, final_time, prediction_horizon+1)
    # plot_results(time, simX[:,0:3], simU)
    # plot_results_2d_3d(time, simX[:,0:3], simU, sphere_radius, sphere_center, start_pos, end_pos)

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
