import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from activations_functional import (relu, relu_prime,
                                    sigmoid, sigmoid_prime,
                                    tanh, tanh_prime,
                                    softmax, softmax_prime
                                      )

import numpy as np
import matplotlib.pyplot as plt

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

# Define the functions
# def f(x):
#     return x**2
#
# def g(x):
#     return np.sin(x)
#
# # Define the range of x values
# x_range = (-10, 10)
#
# # Plot the functions and save the plot to a file
# plot_functions(f, g, x_range, f_name='f(x) = x^2', g_name='g(x) = sin(x)', filename='plot.png')

x_range = (-5, 5)
activations = [(relu, relu_prime, 'relu'), (tanh, tanh_prime, 'tanh'), (sigmoid, sigmoid_prime, 'sigmoid')]
for f, f_prime, f_name in activations:
    plot_functions(f, f_prime, x_range, f_name,
                   f'{f_name}_prime',
                   filename=f'{f_name} and {f_name}_prime.png',
                   show_plot=False)