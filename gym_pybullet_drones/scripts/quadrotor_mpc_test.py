from acados_template import AcadosOcp, AcadosOcpSolver
from quadrotor_model import export_quadrotor_model
from casadi import *
import numpy as np

# Load the quadrotor model (assuming your model function is in export_quadrotor_model)
model = export_quadrotor_model()

# Create the OCP problem
ocp = AcadosOcp()

# Set model in the OCP
ocp.model = model

# Define time discretization parameters
N = 20  # Number of steps (discretization)
T = 1.0  # Time horizon
dt = T / N  # Time step

# Set up the cost function (typically for tracking and control effort)
Q = np.eye(13)  # State cost
R = np.eye(4)   # Control cost

# Set the QP cost for the stage cost (not terminal cost)
ocp.cost.W = np.block([[Q, np.zeros((13, 4))], [np.zeros((4, 13)), R]])  # Stage cost

# Set initial state cost (penalty for initial state)
ocp.cost.W_0 = np.eye(13)  # Penalize deviation of initial state

# Gradient of the cost w.r.t. the initial state (usually zero for no initial state penalty)
ocp.cost.Vx_0 = np.zeros((13, 1))  # Zero if no specific penalty gradient is needed

# Gradient of the cost w.r.t. the initial control input (usually zero for no initial control penalty)
ocp.cost.Vu_0 = np.zeros((4, 1))  # Zero if no specific penalty gradient is needed

# Set the initial state and final state (setpoints for the problem)
x0 = np.zeros((13, 1))  # Initial state (set to zero or some specific initial condition)
xf = np.zeros((13, 1))  # Final state (set as the desired state)

# Set the initial state and final state boundary conditions
ocp.constraints.lbx = x0  # Set the lower bound for state (initial state)
ocp.constraints.ubx = x0  # Set the upper bound for state (initial state)
ocp.constraints.lbx_e = xf  # Set the final state boundary
ocp.constraints.ubx_e = xf  # Set the final state boundary

# Define the control bounds (you can adjust these based on your control limits)
ocp.constraints.lbu = np.array([-1, -1, -1, -1])  # Example lower bounds for control
ocp.constraints.ubu = np.array([1, 1, 1, 1])  # Example upper bounds for control

# Define the time horizon and number of discretization steps
ocp.solver_options.T = T
ocp.solver_options.N = N

# Initial guess (usually, you can initialize it as a simple guess, such as zero control inputs)
u0 = np.zeros((4, N))  # Initial control guess (constant zero control for example)
x0_guess = np.zeros((13, N))  # Initial state guess (constant zero state)

#print(f"State vector dimension: {ocp.nx}")  # State dimension
#print(f"Control vector dimension: {ocp.nu}")  # Control dimension
print(f"W_0 shape: {ocp.cost.W_0.shape}")
print(f"Vx_0 shape: {ocp.cost.Vx_0.shape}")
print(f"Vu_0 shape: {ocp.cost.Vu_0.shape}")


# You can directly set the initial guess when creating the solver:
solver = AcadosOcpSolver(ocp, 'acados_solver')  # Create solver

print("After solver:")
print(f"W_0 shape: {ocp.cost.W_0.shape}")
print(f"Vx_0 shape: {ocp.cost.Vx_0.shape}")
print(f"Vu_0 shape: {ocp.cost.Vu_0.shape}")

# Set initial guess into the OCP
solver.set_initial_guess(x0_guess, u0)

# Solve the OCP
solver.solve()

# Get the solution
x_sol = solver.get(ocp.model.x)  # Extract the state trajectory
u_sol = solver.get(ocp.model.u)  # Extract the control trajectory

print(f"Optimized state trajectory: {x_sol}")
print(f"Optimized control trajectory: {u_sol}")

import matplotlib.pyplot as plt

# Plot state trajectory
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(x_sol[0, :], label="x position")
plt.plot(x_sol[1, :], label="y position")
plt.plot(x_sol[2, :], label="z position")
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Position [m]")
plt.title("State trajectory")

# Plot control trajectory
plt.subplot(2, 1, 2)
plt.plot(u_sol[0, :], label="u1")
plt.plot(u_sol[1, :], label="u2")
plt.plot(u_sol[2, :], label="u3")
plt.plot(u_sol[3, :], label="u4")
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Control input")
plt.title("Control trajectory")

plt.tight_layout()
plt.show()
