import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Regression:
    def plot_piece(self, a, b, n, sigma):
        x = np.random.uniform(-5, 5, n)
        y = a * x + b + np.random.normal(0, sigma, n)  # y = ax + b + noise
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o', label='Noisy data')
        
        x_line = np.linspace(min(x), max(x), 50)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, 'r-', label=f'True line: y={a}x+{b}')  # Drawing the line
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        return fig, ax  # Returning the figure and ax object
    
    def read_file(self, file_path):
        try: 
            # Load the dataset from the file
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Skip the header
            return data[:, 0], data[:, 1]  # Return X and Y columns
        except FileNotFoundError:
            print(f"Error: The file at path {file_path} was not found.")
            return None, None
        except IOError:
            print(f"Error: Could not read the file at path {file_path}.")
            return None, None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None

    def solve_regression(self, x, y):
        x_reshaped = x.reshape(-1, 1)
        
        # Fit the linear regression model
        model = LinearRegression()
        model.fit(x_reshaped, y)
        
        # Get the slope (a_hat) and intercept (b_hat) from the model
        a_hat = model.coef_[0]
        b_hat = model.intercept_
        
        return a_hat, b_hat

    def linear_regression(self, x, y, a_hat, b_hat, ax=None):
        # Create a new plot if there is no existing plot, else add to the existing one
        if ax is None:
            fig, ax = plt.subplots()
            ax.plot(x, y, 'o')
        else:
            ax.plot(x, y, 'o', alpha=0.5, label='Data points')
        
        # Calculate the regression line
        x_line = np.linspace(min(x), max(x), 50)
        y_line = a_hat * x_line + b_hat
        ax.plot(x_line, y_line, 'g-', label=f'Regression line: Y = {a_hat:.2f}X + {b_hat:.2f}')  # Draw the regression line
        
        ax.legend()
        return ax  # Returning the plot for further usage


# Create an instance of Regression class (testing)
reg = Regression()

# Parameters for dataset
a = 5  
b = 2
n = 50
sigma = 5 

# Create the base plot with the true line and noisy data
fig, ax = reg.plot_piece(a, b, n, sigma)

# Read the dataset and perform regression
file_path = 'dataset.txt'
x, y = reg.read_file(file_path)

if x is not None and y is not None:
    a_hat, b_hat = reg.solve_regression(x, y)
   
    # Add the regression line to the existing plot
    ax = reg.linear_regression(x, y, a_hat, b_hat, ax)

# Show the final plot with all lines
plt.show()