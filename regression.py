from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from random import random
from findiff import FinDiff
import numpy as np
import math

# Calcul des estimateurs des paramètres alpha et beta
beta = lambda x, y: np.cov(x,y)[0][1]/np.var(x)
alpha = lambda x, y: np.mean(y) - beta(x, y)*np.mean(x)

# Sum of squared errors function
d = lambda x,y: sum([(x[i]-y[i])**2 for i in range(12)])

def reg(X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), Y1 = np.array([random()*2 for i in range(12)]), Y2 = np.array([random()*2 for i in range(12)])):
    # Création des deux fonctions f1 et f2 alpha + beta * Xi de la regression simple
    f1 = lambda x:alpha(X, Y1) + beta(X, Y1)*x
    f2 = lambda x:alpha(X, Y2) + beta(X, Y2)*x
    
    
    dx1 = 1
    d_dx1 = FinDiff(0, dx1, 1)
    result_f1 = d_dx1(f1(X))
    result_f2 = d_dx1(f2(X))
    print(result_f1[0], result_f2[0])
    
    return 0

reg()