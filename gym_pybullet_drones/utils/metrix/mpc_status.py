"""
    Drone MPC planner project
    About:
        This is a small template to understand how to check solver status,
        which can return 'optimal' for successful solution, 'infeasible' for
        no solution for given constraints and 'unbounded' if there are not
        enough constraints to restrict solution space. Note that 'unbounded'
        MUST always be handled.
            
    Author:
        Adomas Grigolis
        a.grigolis@student.tudelft.nl
        TU Delft 2024
"""

import numpy as np

def mpc_status(status):
    # This function simply converts status codes to integers for easier
    # handling
    converted_status = None
    if status == "optimal": converted_status = 0
    elif status == "infeasible": converted_status = 1
    elif status == "unbounded": converted_status = 2
    else: converted_status = 3
    return converted_status

if __name__ == "__main__":
    # Tests
    timesteps = 100
    problem = None # This will be your CVXPY problem
    stored_results = np.full((timesteps), np.nan)
    
    # In simulation loop
    # problem.solve()
    # stored_results = mpc_status(problem.status)
    stored_results[0] = mpc_status("optimal")
    # Or directly for plotting
    # stored_results = np.full((timesteps, 2), np.nan)
    # stored_results[0, :] = [mpc_status("optimal"), current_timestep]
        
    print(stored_results[0])
    
