import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

#Q1a
def euler(f, t, b, y, h):
    '''
    Eulers method: inputs are the function, intial value for t
    , stopping value b, intial value y0, and the step h.
    '''
    res = []
    while round(t, 2) <= b:
        res.append(y)
        y += h * f(t, y) #Explicit eulers formula
        t += h
    return res


def exactYValues(f,start, end, h):
    '''
    Calculating the points for ploting the graph for the true/exact function.
    '''
    res = []
    while round(start, 2) <= end:
        res.append(f(start))
        start += h
    return res


def tRange(t, step, end):
    '''
    Function to keep track of t's used for plotting.
    '''
    res = []
    while round(t, 2) <= end:
        res.append(t)
        t += step
    return res


def yPrime(t,y):
    return (1/t**2)-(y/t)-y**2


def exactSolution(t):
    return -1/t


t0, y0, h, b = 1, -1, 0.05, 2
yValuesEuler = euler(yPrime, t0, b, y0, h)
yValuesExact = exactYValues(exactSolution,t0,b,h)
tValues = tRange(t0, h, b)


# Q1b
def linear(x,y,xq):
    '''
    Linear interpolation using linear splines formula.
    '''
    # Initiating the resulting vector
    s = [0.0]*(len(xq))
    # We find the range which x is in and use the formula
    for i in range(len(xq)):
        for j in range(len(x)-1):
            if xq[i] >= x[j] and xq[i] <= x[j+1]:
                # m is slope between two points
                m = (y[j+1]-y[j])/(x[j+1]-x[j])
                s[i] = y[j] + m * (xq[i]-x[j])
    return s


points2Compare = [1.052, 1.555, 1.978]
exactInterp = linear(tValues,yValuesExact,points2Compare)
eulerInterp = linear(tValues,yValuesEuler,points2Compare)


def printComparison(methodInterp, q, t):
    print(f'\n{q}')
    print(f'Y(t): {points2Compare}')
    print(f'{t}: {[round(x, 7) for x in methodInterp]}')
    print(f'exact: {[round(x, 7) for x in exactInterp]}')
    print(f'error: {[round(abs(exactInterp[i] - methodInterp[i]), 7) for i in range(len(exactInterp))]}')


printComparison(eulerInterp, 'Question 1b:', "euler")


# Q1c
def modEuler(f, t, b, x, h):
    res = []
    while round(t, 2) <= b:
        res.append(x)
        x += h*f(t, x) + 0.5*h**2*(ft(f,t,x) + fx(f,t,x)*f(t,x))
        t += h
    return res


def ft(f, t, x):
    h = 10**-10
    return (f(t + h, x) - f(t, x))/h


def fx(f, t, x):
    h = 10**-10
    return (f(t, x + h) - f(t, x))/h


yValuesModEuler = modEuler(yPrime, t0, b, y0, h)


# Q1d
modEulerInterp = linear(tValues,yValuesModEuler,points2Compare)
printComparison(modEulerInterp, 'Question 1d:', "modEuler")


# Q1e
def rungeKutta(f, t, b, x, h):
    res = []
    while round(t, 2) <= b:
        res.append(x)
        k1 = h*f(t, x)
        k2 = h*f(t+(h/2), x+(k1/2))
        k3 = h*f(t+(h/2), x+(k2/2))
        k4 = h*f(t+h, x+k3)
        x += (k1/6) + (k2/3) + (k3/3) + (k4/6)
        t += h
    return res


yValuesRK = rungeKutta(yPrime, t0, b, y0, h)


# Q1f
rkInterp = linear(tValues,yValuesRK,points2Compare)
printComparison(rkInterp, 'Question 1f:', "RK")



# Q2
#adamsBashforth 2 Method
def adamsBashforth(f, t, b, y, h):
    yVec = []
    while round(t, 2) <= b:
        yVec.append(y)
        if len(yVec) <= 1:
            y += h * f(t, y) #Using eulers method for initial values
        else:
            y += h * (3*f(t, y) - f(t-h, yVec[-2]))/2
        t += h
    return yVec


#adamsBashforth 3 Method
def adamsBashforth3(f, t, b, y, h):
    yVec = []
    while round(t, 2) <= b:
        yVec.append(y)
        if len(yVec) <= 2:
            y += h * f(t, y) #Using eulers method for initial values
        else:
            y += h * (23*f(t, y) - 16*f(t-h, yVec[-2]) + 5*f(t-2*h, yVec[-3]))/12
        t += h
    return yVec


#adamsMoulton 3 Method
def adamsMoulton3(f, t, b, y, h):
    yVec = []
    while round(t, 2) <= b:
        yVec.append(y)
        if len(yVec) <= 2:
            y += h * f(t, y) #Using eulers method for initial values
        else:
            # Using adamsBashforth to find f(t_n+1, u_n+1)
            w = y + h * (3*f(t, y) - f(t-h, yVec[-2]))/2
            fnPlusOne = f(t+h, w)
            #Adam moultons method
            y += h * (5*fnPlusOne + 8*f(t, y) - 1*f(t-h, yVec[-2]))/12
        t += h
    return yVec

def yPrime2(x,y):
    return y - x**2


def exactSolution2(x):
    return 2 + 2*x + x**2 - math.exp(x)


t0, y0, h, b = 0, 1, 0.1, 3.3
yRK = rungeKutta(yPrime2, t0, b, y0, h)
yExact = exactYValues(exactSolution2,t0,b,h)
xValues = tRange(t0, h, b)
yAdamsBashforth = adamsBashforth3(yPrime2, t0, b, y0, h)
yAdamsMoulton = adamsMoulton3(yPrime2, t0, b, y0, h)

print('\nQ2')
print(f'Runge Kutta 4 Y(3.3) = {yRK[-1]}')
print(f'Adam Bashforth Y(3.3) = {yAdamsBashforth[-1]}')
print(f'Adam Moulton Y(3.3) = {yAdamsMoulton[-1]}')
print(f'Exact value Y(3.3) = {yExact[-1]}')


# Plot --------------------------------------
def plotPoints(X, Y, c):
    '''
    Function for plotting all points used in creating a graph.
    '''
    for i in range(len(X)):
        plt.plot(X[i],Y[i],'ro', color=c)

def myPlot(tValuesMethod, yValuesMethod, solveLinearMethod, t1, t2, l):
    plt.subplot(121)
    methodCol, exactCol = "red", "green"
    plt.plot(tValues, yValuesMethod, color = methodCol)
    method_legend = mpatches.Patch(color=methodCol, label=l)
    plt.plot(tValues, yValuesExact, color=exactCol)
    exact_legend = mpatches.Patch(color=exactCol, label='Exact')
    plt.legend(handles=[method_legend, exact_legend])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title(t1)

    plt.subplot(122)
    plt.plot(points2Compare, solveLinearMethod, color = methodCol)
    method_legend = mpatches.Patch(color=methodCol, label=l)
    plotPoints(points2Compare, solveLinearMethod, methodCol)
    plt.plot(points2Compare, exactInterp, color=exactCol)
    exact_legend = mpatches.Patch(color=exactCol, label='Exact')
    plotPoints(points2Compare, exactInterp, exactCol)
    plt.legend(handles=[method_legend, exact_legend])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title(t2)
    plt.show()

print("\nClose the figures to get the graphs for next problems.")
# 1a and 1b
myPlot(tValues, yValuesEuler, eulerInterp, '1a) Euler\'s method', '1b) y(1.052) y(1.555) y(1.978)', 'Euler')
# 1c and 1d
myPlot(tValues, yValuesModEuler, modEulerInterp, '1c) Modified Euler\'s method', '1d) y(1.052) y(1.555) y(1.978)', 'Modified Euler')
# 1e and 1f
myPlot(tValues, yValuesRK, rkInterp, '1e) RK\'s method', '1f)  y(1.052) y(1.555) y(1.978)', "RK4")


yAdamsBashforth
rkCol, adam1Col, adam2Col, exactCol = "red", "orange", "blue", "green"
plt.plot(xValues, yRK, color = rkCol)
rk_legend = mpatches.Patch(color=rkCol, label='RK4')
plt.plot(xValues, yAdamsBashforth, color = adam1Col)
adam1_legend = mpatches.Patch(color=adam1Col, label='Adams Bashforth')
plt.plot(xValues, yAdamsMoulton, color = adam2Col)
adam2_legend = mpatches.Patch(color=adam2Col, label='Adams Moulton')
plt.plot(xValues, yExact, color=exactCol)
exact_legend = mpatches.Patch(color=exactCol, label='Exact')
plt.legend(handles=[rk_legend,adam1_legend,adam2_legend,exact_legend])
plt.ylabel('y')
plt.xlabel('x')
plt.title('Q2')
plt.show()
