import numpy as np
from sympy import Symbol, sympify, ratsimp, zoo
from os import startfile
from sys import exit
from copy import copy
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d

# Input elements of set
def inputall():
    answer = ''
    while answer != 'f' and answer != 'm':
        answer = input(" Do you want to create set (x[i] > x[i - 1]) from a file or manually? [f/m]: ")
    if answer == 'm':
        print("=====================================================")
        print(" Enter the elements (float) for the set. ")
        for i in range(n):
            while True:
                try:
                    Xs[i] = float(input(" Enter the element x[" + str(i + 1) + "]: "))
                    Ys[i] = f.subs(x, Xs[i])
                    if i > 0 and Xs[i] <= Xs[i - 1]:
                        raise ValueError()
                except ValueError:
                    continue
                else:
                    break
    else:
        # Read a matrix
        with open('input.txt') as fil:
            plot = fil.read()
            plot = plot.split(sep=',')
        if len(plot) == n:
            for i in range(n):
                try:
                    Xs[i] = float(plot[i])
                    Ys[i] = f.subs(x, Xs[i])
                    if i > 0 and Xs[i] <= Xs[i - 1]:
                        raise ValueError()
                except ValueError:
                    print(" There is some invalid element in set. ")
                    input()
                    startfile('input.txt')
                    exit()
        else:
            print(" A number of elements doesn't fit to these you've entered. ")
            input()
            startfile('input.txt')
            exit()


# Output elements of matrix
def outputall():
    print("=====================================================")
    print('Created set of x: {}'.format(Xs))
    print('Created set of y: {}'.format(Ys))
    print("=====================================================")


# Seeking for Lagrange polynomial
def lagrange():
    polynom = 0  # Polynomial
    for i in range(n):
        element = Ys[i]  # Element of one sum
        for k in range(n):
            if i != k:
                element *= (x - Xs[k]) / (Xs[i] - Xs[k])
        polynom += element

    polynom = ratsimp(polynom)  # Simplify the function
    print(polynom)
    return polynom


# Seeking for Newton polynomial
def newton():
    polynom = 0  # Polynomial
    for i in range(n):
        if i == 0:
            element = f.subs(x, Xs[0])  # If it's the first element, which can't be multiplied like others
        else:
            asd = [Xs[j] for j in range(i + 1)]
            element = findmult(asd)
            for k in range(i):
                element *= (x - Xs[k])
        polynom += element

    polynom = ratsimp(polynom)  # Simplify the function
    print(polynom)
    return polynom


# Multiplying different number of equations (sophisticated calculations)
def findmult(asd):
    if len(asd) == 2:
        return (f.subs(x, asd[1]) - f.subs(x, asd[0])) / (asd[1] - asd[0])
    else:
        xk, xi = copy(asd), copy(asd)  # Create a copy of inout list to delete elements and continue a recursion
        del xk[0]
        del xi[-1]
        return (findmult(xk) - findmult(xi)) / (asd[-1] - asd[0])
    # We actually need recursion above to multiply differences (see 'divided differences' for more)


# Cubic polynomial interpolation
def spline():
    # Define Xs and Ys as numpy array and assign 'n' a local value
    xn = np.asarray(Xs)
    yn = np.asarray(Ys)
    n = len(xn) - 1
    # Define h
    hn = np.array([xn[i + 1] - xn[i] for i in range(n)])
    # Create matrix (A) and a vector (b) to solve the system
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Calculate the matrix
    A[0, 0] = 2 * (hn[0] + hn[n - 1])
    A[0, 1] = hn[0]
    A[0, -1] = hn[n - 1]
    b[0] = 6 * ((yn[1] - yn[0]) / hn[0] - (yn[n] - yn[n - 1]) / hn[n - 1])
    for i in range(1, n):
        A[i, i] = 2 * (hn[i - 1] + hn[i])
        A[i, i - 1] = hn[i - 1]
        if i < n - 1:
            A[i, i + 1] = hn[i]
        b[i] = 6 * ((yn[i + 1] - yn[i]) / hn[i] - (yn[i] - yn[i - 1]) / hn[i - 1])
    A[-1, 0] = hn[n - 1]

    # Solve the system that we got (it's not the original that we've got as final result)
    solv = np.linalg.solve(A, b)
    solv = np.append(solv, solv[0])

    # Calculate a, b, c, d
    a = yn
    b = np.array([(yn[i + 1] - yn[i]) / hn[i] - (2. * solv[i] + solv[i + 1]) * hn[i] / 6 for i in range(len(hn))])
    c = solv * 0.5
    d = np.array([(solv[i + 1] - solv[i]) / hn[i] / 6 for i in range(len(hn))])

    print(' Polynomials: ')
    sx = []
    for i in range(n):
        # Create a polynomial and print it
        sx.append(ratsimp(a[i] + b[i] * (x - xn[i]) + c[i] * (x - xn[i]) ** 2 + d[i] * (x - xn[i]) ** 3))
        print('[{}, {}]: {}'.format(Xs[i], Xs[i + 1], sx[i]))

    # Show a plot of all calculations
    plot(sx)


# Creating a plot (necessary)
def plot(sx):
    # step = 0.01
    # vals = np.arange(Xs[0], Xs[-1], step)
    # real = np.array([f.subs(x, i) for i in vals])
    interp = np.zeros(0)
    for i in range(n - 1):
        con = 0
        while Xs[i] + con <= Xs[i + 1]:
            interp = np.append(interp, sx[i].subs(x, Xs[i] + con))
            con += step

    # Combine lists into list of tuples
    points = zip(Xs, Ys)
    # Sort list of tuples by x-value
    points = sorted(points, key=lambda point: point[0])
    # Split list of tuples into two list of x values any y values
    x1, y1 = zip(*points)

    sci_y = sp.interpolate.interp1d(x1, y1, kind='cubic')(vals)  # ScyPy interpolation
    newto = np.array([newt.subs(x, i) for i in vals])  # Newton polynomial calculation
    lagrang = np.array([lag.subs(x, i) for i in vals])  # Lagrange polynomial calculation

    try:
        plt.figure()  # Create a figure (plot)
        plt.xlabel('x')  # Set lbel for x axis
        plt.ylabel('y')  # Set label for y axis
        plt.plot(vals, real, vals, interp, '--', vals, sci_y, vals, newto, vals, lagrang)  # Create a plot
        plt.legend(['Real', 'Self-made interpolation', 'SciPy cubic interpolation', 'Newton polynomial', 'Lagrange polynomial'])  # Sign everything on a plot
        plt.title('Real and interpolated curves of function')  # Define a title
        plt.show()  # Finally, show it
    except ValueError:
        print('-' * 20)
        print(" Unknown error was occured and plot can't be built." +
        "\n Sometimes it's occured by non-adequate input knots. ")


# Start a program
print(" Interpolation polynomials."
      + "\n=====================================================")

n = '0'  # Variable of user input
# Check if a user enters correct data
while True:
    try:
        n = int(input(" Enter a number of cases (integer; > 1): "))
        if n < 2:
            raise ValueError()

        print("=====================================================")
        ans = 0
        while ans != 'f' and ans != 'm':
            ans = input(' Do you want to read a function from a file or to define it manually? [f/m]: ')
        # Create a function
        x = Symbol('x')  # Our symbol (x), which is a variable
        if ans == 'm':
            f = sympify(input(' Enter function: '))  # Our function
        else:
            with open('function.txt', 'r') as file:
                f = sympify(file.read())
        print(' Function is: {}'.format(f))
    except ValueError:
        print(" You've entered wrong number. "
              + "\n=====================================================")
        continue
    except SyntaxError:
        print(" Wrong function was introduced. "
              + "\n=====================================================")
        continue
    else:
        break
print("=====================================================")


Xs = [None] * n
Ys = [None] * n

inputall()
outputall()

# Calculate detailed function values and define if function is unceasing on the interval
# If it is not, it's impossible to solve the equations
step = 0.01  # Step for calculations of function
vals = np.arange(Xs[0], Xs[-1], step)  # Values for function calculations
real = np.array([f.subs(x, i) for i in vals])  # Values of calculated function
if zoo in real:
    print(' Function that you\'ve defined is not unceasing on the defined interval. ')
    opfunc = input(' Do you want to open file with a function?[y/n]: ')
    if opfunc == 'y':
        startfile('function.txt')
    exit()

print(' Let\'s find the Lagrange polynomial: ')
lag = lagrange()
print("=====================================================")
print(' Let\'s find the Newton polynomial: ')
newt = newton()
print("=====================================================")
print(' Let\'s find cubic splines. ')
spline()

print("=====================================================")
