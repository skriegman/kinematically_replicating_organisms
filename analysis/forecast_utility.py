import matplotlib.pyplot as plt

import numpy as np

import sympy

from fastsr.estimators.symbolic_regression import SymbolicRegression

from fastgp.algorithms.fast_evaluate import fast_numpy_evaluate
from fastgp.parametrized.simple_parametrized_terminals import get_node_semantics

from sklearn.model_selection import train_test_split


# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

SEED = 0

# lights per level:
y = [3, 5, 8, 12, 17, 30, 39, 51, 69, 91, 111, 134,
     162, 194, 227, 256, 296, 332, 380, 428, 488,
     542, 618, 680, 743, 839, 932, 1036, 1149, 1282,
     1411, 1557, 1694, 1815, 1947, 2097, 2263, 2428,
     2592, 2774, 2949, 3149, 3373, 3581, 3797, 4013,
     4187, 4381, 4580, 4802, 5024]

X = np.arange(1, len(y)+1)  # level number

quadratic_fit = np.polyfit(X, y, 2)#, w=np.sqrt(y))
# print(quadratic_fit)

print(polyfit(X, y, 2))

exit()

X0 = sympy.Symbol('X0', real=True)

for SEED in range(5, 10):

     np.random.seed(SEED)

     # X_train, X_test, y_train, y_test = train_test_split(X, np.array(y))
     X_train, y_train = X_test, y_test = X, np.array(y)

     sr = SymbolicRegression(ngen=5000, pop_size=1000, seed=SEED)
     sr.fit(X_train, y_train)
     score = sr.score(X_test, y_test)
     # print('Score: {}'.format(score))
     # print('Best Individuals:')
     # sr.print_best_individuals()

     sr.save("seed{}".format(SEED))

     raw_expr_str = str(sr.best_individuals_[0])
     raw_expr_str = raw_expr_str.replace('add', 'Add').replace('multiply', 'Mul')
     expr = sympy.sympify(raw_expr_str)

     print('seed{0}:  {1}'.format(SEED, expr))
