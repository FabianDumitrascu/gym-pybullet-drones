"""
    Drone MPC planner project
    About:
        This is a frontend wrapper for MPC metric utility functions.
        Only call front facing functions from within this file unless you
        know exactly why you need to access low-level functions.
    Current development:
        Currently, all functions are inside the same file. Later they will
        be moved out and abstracted so that only front facing functions are
        accessible from this file.
        Features todo:
            Single RMSE (Done)
            Simplified drone model RSME
            
    Author:
        Adomas Grigolis
        a.grigolis@student.tudelft.nl
        TU Delft 2024
"""
from time import process_time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def get_time():
    # Returns the current clock time
    return process_time()

def time_diff(start, end, decimals=2):
    # Returns the time difference in ms (float). Rounded.
    return round((end - start) * 1000.0, decimals)

def get_collisions(positions, obstacles):
    """
    Check if positions are inside obstacles and calculate depth.
    NOTE: I have no idea how this scales with large datasets.

    Parameters:
    - positions: (N, 3) numpy array of positions over time [x, y, z].
    - obstacles: (O, V, 3) numpy array of obstacles, where O is the number of obstacles,
                 V is the number of vertices, and each vertex has [x, y, z] coordinates.

    Returns:
    - depth: (N, O) numpy array of depths into the obstacles (0 if outside).
    """
    num_positions = positions.shape[0]
    num_obstacles = obstacles.shape[0]

    # Initialize results
    depth = np.zeros((num_positions, num_obstacles))

    # Loop through each obstacle
    for i in range(num_obstacles):
        vertices = obstacles[i]

        # Create a convex hull for the obstacle
        hull = ConvexHull(vertices)

        # Loop through each position
        for j, pos in enumerate(positions):
            # Check if the position is inside the obstacle
            is_inside = all(np.dot(eq[:-1], pos) + eq[-1] <= 0 for eq in hull.equations)

            if is_inside:
                # Calculate depth (minimum distance to the obstacle's surface)
                distances = [
                    abs(np.dot(eq[:-1], pos) + eq[-1]) / np.linalg.norm(eq[:-1])
                    for eq in hull.equations
                ]
                depth[j, i] = min(distances)

    return depth

# Experimental
def get_collisions_bbox(positions, obstacles, drone_dimensions=[0, 0, 0]):
    """
    Check if positions are inside obstacles and calculate depth considering
    drone dimensions.
    Function will return the min penetration depth, i.e., the smallest
    depth value.

    Parameters:
    - positions: (N, 3) numpy array of positions over time [x, y, z].
    - obstacles: (O, V, 3) numpy array of obstacles, where O is the number of obstacles,
                 V is the number of vertices, and each vertex has [x, y, z] coordinates.
    - drone_dimensions: List of [width, height, depth] of the drone's bounding box.

    Returns:
    - depth: (N, O) numpy array of depths into the obstacles (0 if outside).
    """
    num_positions = positions.shape[0]
    num_obstacles = obstacles.shape[0]

    # Initialize results
    depth = np.zeros((num_positions, num_obstacles))

    # Calculate half dimensions of the drone
    half_width, half_height, half_depth = [dim / 2 for dim in drone_dimensions]

    # Loop through each obstacle
    for i in range(num_obstacles):
        vertices = obstacles[i]

        # Create a convex hull for the obstacle
        hull = ConvexHull(vertices)

        # Loop through each position
        for j, pos in enumerate(positions):
            # Create the bounding box for the drone around the center position
            drone_min_bounds = pos - np.array([half_width, half_height, half_depth])
            drone_max_bounds = pos + np.array([half_width, half_height, half_depth])

            # Check each corner of the drone's bounding box
            corners = [
                drone_min_bounds,
                drone_max_bounds,
                [drone_min_bounds[0], drone_min_bounds[1], drone_max_bounds[2]],
                [drone_min_bounds[0], drone_max_bounds[1], drone_min_bounds[2]],
                [drone_max_bounds[0], drone_min_bounds[1], drone_min_bounds[2]],
                [drone_min_bounds[0], drone_max_bounds[1], drone_max_bounds[2]],
                [drone_max_bounds[0], drone_min_bounds[1], drone_max_bounds[2]],
                [drone_max_bounds[0], drone_max_bounds[1], drone_min_bounds[2]],
            ]

            is_inside = any(
                all(np.dot(eq[:-1], corner) + eq[-1] <= 0 for eq in hull.equations)
                for corner in corners
            )

            if is_inside:
                # Calculate depth (minimum distance to the obstacle's surface)
                distances = [
                    min(
                        abs(np.dot(eq[:-1], corner) + eq[-1]) / np.linalg.norm(eq[:-1])
                        for corner in corners
                    )
                    for eq in hull.equations
                ]
                depth[j, i] = min(distances)

    return depth

def get_rmse(ground_truth, values):
    """
    Parameters
    ----------
    ground_truth : numpy ndarray
        True value.
    values : numpy ndarray
        Data points.

    Raises
    ------
    TypeError
        If inputs provided not as numpy ndarrays.
    ValueError
        If inputs not the same shape.

    Returns
    -------
    rmse : numpy ndarray
        Returns numpy array shape (n,) where n is number of columns.
        Each value represents RMSE over each data column.

    """
    # Type checks
    if not isinstance(ground_truth, np.ndarray):
        raise TypeError("'ground_truth' must be a NumPy array.")
    if not isinstance(values, np.ndarray):
        raise TypeError("'values' must be a NumPy array.")
    if ground_truth.shape != values.shape:
        raise ValueError("Numpy arrays must be the same shape.")
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((ground_truth - values) ** 2, axis=0))
    return rmse

def generic_multiline_plotter(
        data_list,
        labels=[], title="Line Graph",
        xlabel="Time", ylabel="Value",
        show_grid=True):
    """
    Plots multiple line graphs from a list of NumPy arrays with specified labels.
    
    Parameters:
    - data_list: List of NumPy arrays, each array is a set of y-values to plot.
    - labels: List of labels corresponding to each line (default: []).
    - title: The title of the graph (default: "Line Graph").
    - xlabel: Label for the X-axis (default: "time").
    - ylabel: Label for the Y-axis (default: "Value").
    - show_grid: Shows grid (default: True).
    """
    # Check if provided data is not added into a list
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]
    # Check if legends provided properly
    if len(data_list) != len(labels) and not labels == []:
        raise ValueError("The number of data arrays must match the number of labels.")
    
    # Plotter
    for data in data_list:
        plt.plot(data[:, 0], data[:, 1])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels != []:
        plt.legend(labels)
    plt.grid(show_grid)
    plt.show()

if __name__ == "__main__":
    # Tests
    st = get_time()
    
    gt = np.random.rand(100, 2)
    val = np.random.rand(100, 2)
    rmse = get_rmse(gt, val)
    
    positions = np.array([
        [0.5, 0.5, 0.5],  # Inside the first obstacle
        [1.5, 1.5, 1.5],  # Outside both obstacles
        [2.5, 2.5, 2.5],  # Inside the second obstacle
        [3.1, 3.1, 3.1]   # Outside both obstacles
    ])

    # Obstacles: (O, V, 3) numpy array
    obstacles = np.array([
        # First obstacle: A cube from (0,0,0) to (1,1,1)
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        
        # Second obstacle: A cube from (2,2,2) to (3,3,3)
        [[2, 2, 2], [3, 2, 2], [3, 3, 2], [2, 3, 2], [2, 2, 3], [3, 2, 3], [3, 3, 3], [2, 3, 3]]
    ])
    
    # Check positions
    depth = get_collisions(positions, obstacles)
    print("Depth:")
    print(depth)
    
    depth = get_collisions_bbox(positions, obstacles, [0.1,0.1,0.1])
    print("Depth:")
    print(depth)
    
    et = get_time()
    print(rmse)
    print("Computation time: ", time_diff(st, et), " ms")
