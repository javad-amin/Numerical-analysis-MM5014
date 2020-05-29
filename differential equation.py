import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import sqrt, pi, cos, sin, exp

#a
def euler(f, t, b, u, n):
    '''
    Eulers method: inputs are the function, intial t value
    , stopping value b, intial value u(0), number of points.
    '''
    h = (b-t)/n
    res = [[],[]]
    while round(t, 2) <= b:
        res[0].append(t)
        res[1].append(u)
        u += h * f(t, u) #Explicit eulers formula
        t += h
    return res


#b
def rk2(f, t, b, u, n):
    '''
    inputs are the function, intial t value
    , stopping value b, intial value u(0), number of points.
    Uses the Modified euler method.
    '''
    h = (b-t)/n
    res = [[],[]]
    while round(t, 2) <= b:
        res[0].append(t)
        res[1].append(u)
        u += h*f(t, u) + 0.5*h**2*(ft(f,t,u) + fu(f,t,u)*f(t,u))
        t += h
    return res


def ft(f, t, x):
    h = 10**-10
    return (f(t + h, x) - f(t, x))/h


def fu(f, t, x):
    h = 10**-10
    return (f(t, x + h) - f(t, x))/h


#c
def rk4(f, t, b, u, n):
    '''
    Runge kutta 4th order method: inputs are the function, intial t value
    , stopping value b, intial value u(0), number of points.
    '''
    h = (b-t)/n
    res = [[],[]]
    while round(t, 2) <= b:
        res[0].append(t)
        res[1].append(u)
        k1 = h*f(t, u)
        k2 = h*f(t+(h/2), u+(k1/2))
        k3 = h*f(t+(h/2), u+(k2/2))
        k4 = h*f(t+h, u+k3)
        u += (k1/6) + (k2/3) + (k3/3) + (k4/6)
        t += h
    return res


#d
def adamsBashforth3(f, t, b, u, n):
    '''
    Adam Bashforth method: inputs are the function, intial t value
    , stopping value b, intial value u(0), number of points.
    '''
    h = (b-t)/n
    res = [[],[]]
    while round(t, 2) <= b:
        res[0].append(t)
        res[1].append(u)
        if len(res[1]) <= 2:
            u += h*f(t, u) + 0.5*h**2*(ft(f,t,u) + fu(f,t,u)*f(t,u)) #Using rk2 for initial values
        else:
            u += h * (23*f(t, u) - 16*f(t-h, res[1][-2]) + 5*f(t-2*h, res[1][-3]))/12
        t += h
    return res


def adamsMoulton3(f, t, b, u, n):
    '''
    Adam Moulton method: inputs are the function, intial t value
    , stopping value b, intial value u(0), number of points.
    '''
    h = (b-t)/n
    res = [[],[]]
    while round(t, 2) <= b:
        res[0].append(t)
        res[1].append(u)
        if len(res[1]) <= 2:
            u += h*f(t, u) + 0.5*h**2*(ft(f,t,u) + fu(f,t,u)*f(t,u)) #Using rk2 for initial values
        else:
            # Using adamsBashforth to find f(t_n+1, u_n+1)
            w = u + h * (3*f(t, u) - f(t-h, res[1][-2]))/2
            fnPlusOne = f(t+h, w)
            #Adam moultons method
            u += h * (5*fnPlusOne + 8*f(t, u) - 1*f(t-h, res[1][-2]))/12
        t += h
    return res


# Exact
def exactValues(f,t, b, n):
    '''
    Calculating the points for ploting the graph for the true/exact function.
    '''
    h = (b-t)/n
    res = [[],[]]
    while round(t, 2) <= b:
        res[0].append(t)
        res[1].append(f(t))
        t += h
    return res


def exactSolution(t):
    '''
    Exact solution
    '''
    return (-cos(pi*t)+pi*sin(pi*t)+(3+2*pi**2)*exp(t))/(1+pi**2)

def getErrors(exact, sol):
    '''
    Getting exact values and the values calculated by some method it
    returns the error at each step.
    '''
    res = [[],[]]
    for i in range(len(exact[0])):
        diff = exact[1][i] - sol[1][i]
        res[0].append(exact[0][i])
        res[1].append(diff)
    return res


# # Plot --------------------------------------
def plotPoints(X, Y, c):
    '''
    Function for plotting all points used in creating a graph.
    '''
    for i in range(len(X)):
        plt.plot(X[i],Y[i],'ro', color=c)

def myPlot(method, t1, yScale):
    print('===Wait for it...===\n===When you done,close the figure to plot another one.===\n')
    def uPrime(t,u):
        '''
        The function we are going to investigate:
        du/dt = cos(Pi*t) + u(t)
        '''
        return cos(pi*t)+u
    plt.subplot(121)
    # Variabels and application of the choosen method.
    t0, tn, u0 = 0, 2, 2
    N = [10, 20, 40, 80, 160, 320, 640]
    NCol = ["black","darkred","red","orange","y","blue","deeppink"]
    legends = []
    for i in range(len(N)):
        valuesMethod = method(uPrime, t0, tn, u0, N[i])
        methodCol = NCol[i]
        plt.plot(valuesMethod[0], valuesMethod[1], color = methodCol)
        legend = mpatches.Patch(color=methodCol, label=N[i])
        legends.append(legend)
    # Drawing the exact values
    exactCol = "green"
    exactV = exactValues(exactSolution, t0,tn,N[-1])
    plt.plot(exactV[0], exactV[1], color=exactCol)
    exact_legend = mpatches.Patch(color=exactCol, label='Exact')
    legends.append(exact_legend)
    plt.legend(handles=legends)
    plt.ylabel('u')
    plt.xlabel('t')
    plt.title(t1)
    # Drawing the errors
    plt.subplot(122)
    for i in range(len(N)):
        valuesMethod = method(uPrime, t0, tn, u0,N[i])
        exactV = exactValues(exactSolution, t0,tn,N[i])
        methodErrors = getErrors(exactV, valuesMethod)
        methodCol = NCol[i]
        plt.plot(methodErrors[0], methodErrors[1], color = methodCol)
    plt.legend(handles=legends[:-1])
    plt.ylabel('u')
    plt.xlabel('t')
    if yScale != "z":
        plt. ylim((-0.1, 2.25)) # Remove this line to change y scale of error
    plt.title("Errors")
    plt.show()


# Getting input from user
while True:
    inp = input("Type the number of the method you want to plot.\nOBS: type z after the number if you want to see error in original scale.\nExample: \"3z\" plots rk4 with zoomed in error sizes otherwise \"3\" plots errors in scale y=2.25.\n\n1. Euler\n2. rk2 \n3. rk4 \n4. Adam Mounton\n0. to exit >>>")
    if inp == "0":
        print("( ^_^)／")
        break
    if inp[0] == "1":
        myPlot(euler, 'Euler\'s method', inp[-1])
    if inp[0] == "2":
        myPlot(rk2, 'RK2 method', inp[-1])
    if inp[0] == "3":
        myPlot(rk4, 'RK4 method', inp[-1])
    if inp[0] == "4":
        myPlot(adamsMoulton3, 'Adam Moulton method', inp[-1])
