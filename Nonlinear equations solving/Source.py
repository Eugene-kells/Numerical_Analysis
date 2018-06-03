import os
import sys
import numpy as np  # To build a data for a plot
import matplotlib.pyplot as plt  # To build a plot
from sympy import div  # To divide polynomials (hatred!!1!)
from sympy import Symbol  # To define equation
from sympy import diff  # To find a derivative (can also do it by hands #) )
from sympy import Poly  # To find a coefficients of a sympy polynomial
from math import ceil, floor


# Input coefficients
def inputall():
    answer = ''
    arr = []
    while answer != 'f' and answer != 'm':
        answer = input(" Do you want to read coefficients from a file or enter manually? [f/m]: ")
    if answer == 'm':
        while True:
            try:
                print("=====================================================")
                n = int(input(' Enter the number of equation\'s elements (integer; > 0): '))
                if n < 1:
                    raise ValueError()
                print(" Enter the coefficients (float or int) from 0 to n. ")
                arr = [(float(input(' Enter free element: ')))]
                for i in range(1, n):
                    arr.append(float(input(' Enter element [' + str(i) + ']: ')))
                print("=====================================================")
            except ValueError:
                print(" You've entered incorrect value. ")
            else:
                break
    else:
        with open('equation.txt') as file:
            plot = file.read()
            plot = plot.split()
            n = len(plot)  # Length of equation
            # Read a coefficients
            for i in plot:
                try:
                    arr.append(float(i))  # Append elements of a file to our coefficients
                except ValueError:
                    print(" There is some invalid element in coefficients. ")
                    input()
                    os.startfile('equation.txt')
                    sys.exit(0)
    return arr, n


# Define circle for complex numbers
def circ():
    a = max(abs(i) for i in coef[: n - 1])
    b = max(abs(i) for i in coef[1:])
    rez1 = abs(coef[0]) / (b + abs(coef[0]))
    rez2 = (abs(coef[n - 1]) + a) / abs(coef[n - 1])
    return rez1, rez2


# Calculate interval by using formulas
def valks(array):
    au = max(abs(i) for i in array if i < 0)  # Find biggest absolute value in a polynom of x < 0 element
    if array[-1] < 0:
        array = [-i for i in array]
    m = max(array.index(i) for i in array if i < 0)  # Seek position of the last (x < 0) element
    return 1 + (au / array[-1]) ** (1 / (n - m - 1))


# Gua's theorem
def gua():
    for i in range(1, n - 1):
        #print(coef[i], coef[i - 1], coef[i + 1], coef[i] ** 2, coef[i - 1] * coef[i + 1])
        if coef[i] ** 2 <= coef[i - 1] * coef[i + 1]:
            return ' Equation has 2 complex conjugate roots. '
    return ' Equation doesn\'t have 2 complex conjugate roots. '


# Changes in a table
def changes(table, ern=1):
    # Calculate sign changes using created table
    somes = []
    for j in range(ern, len(table)):
        some = 0
        for i in range(1, len(table[0])):
            if (table[j][i - 1] == '-' and table[j][i] == '+') or (table[j][i - 1] == '+' and table[j][i] == '-'):
                some += 1
        somes.append(some)
    return somes


# Deeper analysis of intervals using the same code as in 'sturm' function
def deepanalysis(divid):
    # Creating a table
    results = []
    varibls = np.arange(floor(Rlo), ceil(Rup + 1), 0.1)
    varibls = varibls.tolist()  # Convert to a list
    for i in range(len(varibls)):
        results.append([])
        results[i].append(varibls[i])

    # Calculate value for each function and possible variable
    for j in range(len(varibls)):
        for i in divid:
            val = results[j][0]
            if i.evalf(subs={x: val}) > 0:
                results[j].append('+')
            else:
                results[j].append('-')

    # Check number of sign changes
    ert = changes(results, 0)
    for i in range(len(results)):
        results[i].append(ert[i])
    # for i in results:
    #     print(i)

    send = []
    for i in range(len(results) - 1):
        if results[i][-1] != results[i + 1][-1]:
            temp = [varibls[i] for i in range(len(varibls))]
            send.append((temp[i], temp[i + 1]))
    return send


# Sturm method
def sturm():
    divid = [fx, diff(fx)]  # Array where equations will be stored
    while True:
        test = Poly(divid[-1], x)
        if len(test.coeffs()) > 1:
            q, r = div(divid[-2], divid[-1], x)  # Taking quotient and remainder of dividing
            divid.append(-1 * r)
        else:
            break
    print('\n Let\'s calculate dividing of a polynomials: ')
    for i in divid:
        print('f' + str(divid.index(i)) + '(x): ' + str(i))

    # Create information table
    table = [['   '], ['-inf'], ['+inf']]
    for i in divid:
        table[0].append('f' + str(divid.index(i)) + '(x)')  # Better information association
        test = Poly(i, x)
        # -inf
        if (test.degree() % 2 == 0 and test.coeffs()[0] > 0) or \
                (test.degree() % 2 != 0 and test.coeffs()[0] < 0):
            table[1].append('+')
        else:
            table[1].append('-')
        # +inf
        if test.coeffs()[0] > 0:
            table[2].append('+')
        else:
            table[2].append('-')

    qwe = changes(table)
    mininf = qwe[0]
    plusinf = qwe[1]

    # Add found data to a table for better user experience
    table[0].append('Sign changes')
    table[1].append(mininf)
    table[2].append(plusinf)
    print('---------')
    print('\n\n Then we need to calculate number of sign changes for (-inf) and (+inf) cases. ')
    print(' Let\'s create a table to visualize our results: ')

    for i in table:
        print()
        for j in i:
            if table.index(i) == 0:
                print('  ' + j, end='')
            else:
                print(j, end='     ' + ' ' * i.index(j))
    print('\n\n As we see, our equation has {0} - {1} = {2} real roots.'.format(mininf, plusinf, mininf - plusinf))

    print('---------')
    print('\n\n Now we need to define possible intervals for roots. ')
    print(' Let\'s create a table with calculated functions in possible root interval. ')

    # Creating a table
    results = [['    ']]
    for i in divid:
        results[0].append('f' + str(divid.index(i)) + '(x) ')  # Better information association
    for i in range(floor(Rlo), ceil(Rup + 1)):
        results.append(['x = ' + str(i)])

    # Calculate value for each function and possible variable
    count = 1
    for j in range(floor(Rlo), ceil(Rup + 1)):
        for i in divid:
            if i.evalf(subs={x: j}) > 0:
                results[count].append('+')
            else:
                results[count].append('-')
        count += 1

    # Check number of sign changes
    ert = changes(results)
    results[0].append('Sign changes')
    for i in range(1, len(results)):
        results[i].append(ert[i - 1])

    # Printing table
    for i in results:
        print()
        for j in i:
            if results.index(i) == 0:
                print('  ' + j, end='')
            else:
                print(j, end='     ' + ' ' * i.index(j))

    send = []
    for i in range(1, len(results) - 1):
        if results[i][-1] != results[i + 1][-1]:
            temp = [i for i in range(floor(Rlo), ceil(Rup + 1))]
            send.append((temp[i - 1], temp[i]))

    # Check if we need extended computations
    analys = False
    for i in range(1, len(send)):
        if send[i][0] == send[i - 1][1]:
            analys = True
            break
    if len(send) != mininf - plusinf or analys is True:
        print('\n\nWARNING!!! PROGRAM CALCULATE VALUES FROM {0} TO {1} with step 0.1. IT\'S NOT SHOWN IN A TABLE.'.format(floor(Rlo), ceil(Rup + 1))
              + '\nIT HAPPENS BECAUSE TWO OR MORE ROOTS LIE IN INTERVAL WITH SIZE LESS THAN 1'
              + '\nOR THEIR INTERVALS STARTS AND BEGINS AT ONE POINT RESPECTIVELY.'
              + '\nWITHOUT THIS CHECKING CALCULATIONS WILL BE WRONG.')
        send = deepanalysis(divid)

    print('\n As we can see, roots of equation lie in intervals: ')
    for i in send:
        print(i)

    return send  # Return intervals of roots


# Bisection method
def bisection(a, b):
    if abs(fx.evalf(subs={x: a})) < e:
        return a
    elif abs(fx.evalf(subs={x: b})) < e:
        return b
    c = (a + b) / 2
    # Let's use loop instead of recursion to avoid stack overflow
    # Check where roots are
    while abs(fx.evalf(subs={x: c})) > e or abs(b - a) > e:
        c = (a + b) / 2
        if fx.evalf(subs={x: a}) * fx.evalf(subs={x: c}) < 0:
            a, b = a, c
        else:
            a, b = c, b
    return (a + b) / 2  # Return middle of our found small interval (it'll be very well-defined number)


# Chords method
def chords(a, b):
    if abs(fx.evalf(subs={x: a})) < e:
        return a
    elif abs(fx.evalf(subs={x: b})) < e:
        return b
    c = (a * fx.evalf(subs={x: b}) - b * fx.evalf(subs={x: a})) / (fx.evalf(subs={x: b}) - fx.evalf(subs={x: a}))
    # Check where roots are
    while abs(fx.evalf(subs={x: c})) > e:
        c = (a * fx.evalf(subs={x: b}) - b * fx.evalf(subs={x: a})) / (fx.evalf(subs={x: b}) - fx.evalf(subs={x: a}))
        if fx.evalf(subs={x: a}) * fx.evalf(subs={x: c}) < 0:
            a, b = a, c
        else:
            a, b = c, b
    return (a * fx.evalf(subs={x: b}) - b * fx.evalf(subs={x: a})) / (fx.evalf(subs={x: b}) - fx.evalf(subs={x: a}))


# Newton's method
def newton(xk0, a, b):
    if abs(fx.evalf(subs={x: a})) < e:
        return a
    elif abs(fx.evalf(subs={x: b})) < e:
        return b
    equ1 = diff(fx)  # Find first derivative of original polynomial
    xk1 = xk0
    while abs(xk1 - xk0) > e or abs(fx.evalf(subs={x: xk1})) > e:
        xk0 = xk1
        xk1 = xk0 - fx.evalf(subs={x: xk0}) / equ1.evalf(subs={x: xk1})
    return xk1


# Build a plot (just for fun)
def plot():
    xa = np.arange(interv[0][0] - 1, interv[-1][-1] + 1, 0.01)
    y = np.zeros((len(xa)))
    for i in range(len(xa)):
        #y[i] = funcval(x[i], coef)
        y[i] = fx.evalf(subs={x: xa[i]})
    plt.plot(xa, y)
    plt.show()


# Start a program
print("\n\n Solving non-linear equations. "
      + "\n\n We have an equation that looks like this (int general):"
      + "\n f(x) = a(n)x^n + a(n-1)x^(n-1) + ... + a(1)x + a(0) ")

n = '0'  # Variable of user input

# Check if a user enters correct data
while True:
    try:
        interm, n = inputall()
        e = abs(float(input(' Enter precision (in format 0.001...): ')))
        while interm[-1] == 0:
            del interm[-1]
            n -= 1
        coef = tuple(interm)  # Coefficients of equation members
        if coef[-1] < 0:
            coef = tuple(-i for i in coef)
        print(coef)

    except ValueError:
        print(" You've entered something wrong. Check entered data. "
              + "\n=====================================================")
        continue
    else:
        break
print("\n=====================================================")

# Define 'x' as global unknown variable and creating sympy form of original equation
x = Symbol('x')
fx = 0  # First (original) equation
for i in range(len(coef)):
    fx += int(coef[i]) * x ** i


# Calculate circle for complex numbers
print('\n Let\'s theorem about limits of complex roots. ')
circle = tuple(circ())
print(' Circle for absolute complex numbers is: {0}'.format(str(circle)))

# Calculate borders for values
print('===============================')
Rup = valks(coef)  # Upper border
neg = tuple([i if coef.index(i) % 2 == 0 else -i for i in coef])  # Creating -x polynom
Rlo = valks(neg) * -1  # Lower border
print('\n Let\'s use theorem about borders of roots. ')
print(' Roots of equation lie in interval: ({0}, {1})'.format(str(Rlo), str(Rup)))

# Gua theorem
print('===============================')
print('\n Let\'s check Gua\'s theorem. ')
print(gua())

# Let's use Sturm theorem...
print('===============================')
print(' Let\'s use Sturm method: ')
try:
    interv = sturm()
except TypeError:
    print('\n\n Ooops! It looks like that intervals are complex! This application can\'t solve such problem. ')
    input()
    sys.exit(0)


# Let's use method of bisection
print('===============================')
print(' Let\'s use bisection\'s method: ')
for i in range(len(interv)):
    print(' Root X{0} equals to: {1}'.format(i + 1, bisection(interv[i][0], interv[i][1])))

# Let's use method of chords
print('===============================')
print(' Let\'s use chord\'s method: ')
for i in range(len(interv)):
    print(' Root X{0} equals to: {1}'.format(i + 1, chords(interv[i][0], interv[i][1])))

# Let's use Newton's method
print('===============================')
print(' Let\'s use Newton\'s method: ')
for i in range(len(interv)):
    print(' Root X{0} equals to: {1}'.format(i + 1, newton((interv[i][0] + interv[i][1]) / 2, interv[i][0], interv[i][1])))


print('===============================')
input(' That\'s all! Press any button to build a plot for a equation. ')
plot()
