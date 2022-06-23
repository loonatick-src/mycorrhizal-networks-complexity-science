import numpy as np

def hill_equation(k, n):
    def _hill_function(x):
        x_n = np.power(x, n)
        k_n = np.power(k, n)
        return x_n/(k_n + x_n)
    
    return _hill_function
