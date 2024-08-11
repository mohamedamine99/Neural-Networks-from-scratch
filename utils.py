import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def generate_linear_points_with_noise(x_range, slope, intercept, num_points, noise_std_dev):
    """
    Generates a set of points representing a linear function with random noise.

    Parameters:
    x_range (tuple): A tuple (start, end) defining the range of x values.
    slope (float): The slope of the linear function.
    intercept (float): The intercept of the linear function.
    num_points (int): The number of points to generate.
    noise_std_dev (float): The standard deviation of the Gaussian noise.

    Returns:
    x (np.array): The x values.
    y (np.array): The y values with noise.
    """
    # Generate x values
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Generate y values for the linear function
    y = slope * x + intercept

    # Add Gaussian noise to the y values
    noise = np.random.normal(0, noise_std_dev, num_points)
    y_noisy = y + noise

    return x, y_noisy


def plot_functions(f, g, x_range, f_name='func1', g_name='func2', filename=None, show_plot = True):
    """
    Plots two functions on the same graph and optionally saves the plot to a file.

    Parameters:
    f (function): The first function to plot.
    g (function): The second function to plot.
    x_range (tuple): A tuple (start, end) defining the range of x values.
    f_name (str): The name of the first function for the legend.
    g_name (str): The name of the second function for the legend.
    filename (str, optional): The filename to save the plot. If None, the plot is not saved.
    """
    # Define the range of x values
    x = np.linspace(x_range[0], x_range[1], 400)

    # Evaluate the functions
    y1 = f(x)
    y2 = g(x)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label=f_name)
    plt.plot(x, y2, label=g_name)

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Graphs of {f_name} and {g_name}')

    # Add a legend
    plt.legend()

    # Add a grid
    plt.grid(True)

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")

    # Show the plot
    if show_plot:
        plt.show()


def min_max_normalize(arr):
    """
    Min-max normalizes a numpy array.

    Parameters:
    arr (np.array): The input numpy array.

    Returns:
    normalized_arr (np.array): The min-max normalized numpy array.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr
