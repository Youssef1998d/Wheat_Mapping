from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from findiff import FinDiff
import numpy as np
from pixels import observations

def compare(X, v): 
    fig = plt.figure(1)
    fig.subplots_adjust(hspace=0.4, top=0.85)
    fig.suptitle("Comparaison des fonctions Yi")
    ax, type = [], ("Regression Linéaire Simple", "Reression Polynomiale", "Les Nombres Dérivées")
    for i in range(len(v)*3):
        ax.append(fig.add_subplot(len(v),3,i))
        ax[i].set_title(type[i%3])
        ax[i].plot(X, v[i%3][i%3]) 
    plt.show()

def explicate(X, Y, degree_ = 1):
 
    poly_reg = PolynomialFeatures(degree = degree_)
    X_poly = poly_reg.fit_transform(np.array(X).reshape(-1, 1))
    
    lin_reg_1=LinearRegression().fit(X_poly, Y)
    poly = lin_reg_1.predict(X_poly)

    d_dx = FinDiff(0, X[1] - X[0], 1)
    der = d_dx(poly)

    return np.mean(poly), np.mean(der)

def generate_explications(X, Y, degree=2):
    for y in Y:
        yield explicate(degree, X, y)
    
generate_explications([1, 2, 3, 4], observations)
