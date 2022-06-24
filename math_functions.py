import numpy as np

def hill_equation(k, n):
    def _hill_function(x):
        x_n = np.power(x, n)
        k_n = np.power(k, n)
        return x_n/(k_n + x_n)
    
    return _hill_function

def uptake_function(x, A = 1.0, K = 1.0, L = 1.0):
    return  A*x/(K + x*(1 + x/L))
