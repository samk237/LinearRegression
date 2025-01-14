import numpy as np
import matplotlib.pyplot as plt

class Regression:
    def linear_regression(self, a, b, n, sigma):
        x = np.random.normal(0, sigma, n)
        y = a * x + b + np.random.normal(0, sigma, n)  # y = ax + b + noise
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o') #noise
        x_line = np.linspace(min(x), max(x), 100)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, 'r-') #drawing the line
        #saving as a file
        with open('dataset.txt', 'w') as file:
            file.write(str(x) + ',' + str(y) + '\n')
        return plt.show() #returning the plot

reg = Regression()
reg.linear_regression(2, 2, 100, 1) #example method call