import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

#Q1
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

def lagrange(x,y,xq):
    '''
    Polymonial interpolation using lagrange method.
    '''
    s = [0.0]*(len(xq))
    for k in range(len(xq)):
        for i in range(len(x)):
            L = 1.0 # initiating L
            for j in range(len(x)):
                if i != j: # Formula from lecture notes
                    L *= (xq[k]-x[j]) / (x[i]-x[j])
            s[k] += L * y[i]
    return [round(x, 2) for x in s]


x = [1, 2, 3, 4, 6, 8, 10]
y = [2, 2.5, 7, 10.5, 12.75, 13, 13]

def xq(s,e,i):
    '''
    A function to generate the query points from
    start, end points and an increment value.
    '''
    xq = []
    while s <= e:
        xq.append(s)
        s += i
    return xq


def plotPoints(X, Y):
    '''
    Function for plotting all points used in creating a graph.
    '''
    for i in range(len(X)):
        plt.plot(X[i],Y[i],'ro')

myXq = xq(1,10,0.5)
solveLinear = linear(x,y,myXq)
solveLagrange = lagrange(x,y,myXq)
print('Q1')
print(f'linear: {solveLinear}')
print(f'lagrange: {solveLagrange}')



# Q2.
def euler(f, x, b, y, h):
    '''
    Eulers method: inputs are the function, intial value for x
    , stopping value b, intial value y0, and the step h.
    '''
    res = []
    while round(x, 2) <= b:
        res.append(y)
        y += h * f(x, y) #Explicit eulers formula
        x += h
    return res


'''
As the result is showing the error at Y(1)
gets less as the step size gets smaller.
Y(1) with h = 0.1 => 0.703 ||| Error = 0.062
Y(1) with h = 0.05 => 0.673 ||| Error = 0.032
Y(1) with h = 0.01 => 0.648  ||| Error = 0.007
Y(1) with exact value => 0.641

From the figure we also see that at the begining
the error size is small and as the steps grow
so does the error size.
'''

def xRange(x, step, end):
    '''
    Function to keep track of x's used for plotting.
    '''
    res = []
    while round(x, 2) <= end:
        res.append(x)
        x += step
    return res



def f(x,y):
    return y - x


def exactSolution(x):
    return x+1-(1/2)*math.exp(x)


def exactYValues(start, end, h):
    '''
    Calculating the points for ploting the graph for the true/exact function.
    '''
    res = []
    while round(start, 2) <= end:
        res.append(exactSolution(start))
        start += h
    return res


x0, y0, h1, h2, h3, b = 0, 0.5, 0.1, 0.05, 0.01, 1
eulerH1 = euler(f, x0, b, y0, h1)
xRangeH1 = xRange(x0, h1, b)
eulerH2 = euler(f, x0, b, y0, h2)
xRangeH2 = xRange(x0, h2, b)
eulerH3 = euler(f, x0, b, y0, h3)
xRangeH3 = xRange(x0, h3, b)
exact = exactYValues(x0,b,h3)
xRangeExact = xRange(x0, h3, b)

print('\nQ2:')
print(f'Y(1) with h = 0.1 => {round(eulerH1[-1],3)} ||| Error = {round(abs(round(eulerH1[-1],3) - exactSolution(1)),3)}')
print(f'Y(1) with h = 0.05 => {round(eulerH2[-1],3)} ||| Error = {round(abs(round(eulerH2[-1],3) - exactSolution(1)),3)}')
print(f'Y(1) with h = 0.01 => {round(eulerH3[-1],3)}  ||| Error = {round(abs(round(eulerH3[-1],3) - exactSolution(1)),3)}')
print(f'Y(1) with exact value => {round(exactSolution(1),3)}')


#Plotting
#Q1
plt.subplot(121)
plt.plot(myXq, solveLinear)
plotPoints(myXq, solveLinear)
plt.ylabel('y')
plt.xlabel('x')
plt.title('Linear')

plt.subplot(122)
plt.plot(myXq, solveLagrange)
plotPoints(myXq, solveLagrange)
plt.ylabel('y')
plt.xlabel('x')
plt.title('Lagrange')
plt.show()


#Q2
h1Col, h2Col, h3Col, exactCol = "red", "midnightblue", "blue", "green"
plt.plot(xRangeH1, eulerH1, color = h1Col)
h1_legend = mpatches.Patch(color=h1Col, label='h=0.1')

plt.plot(xRangeH2, eulerH2, color = h2Col)
h2_legend = mpatches.Patch(color=h2Col, label='h=0.05')

plt.plot(xRangeH3, eulerH3, h3Col)
h3_legend = mpatches.Patch(color=h3Col, label='h=0.01')

plt.plot(xRangeExact, exact, color=exactCol)
exact_legend = mpatches.Patch(color=exactCol, label='exact')


plt.legend(handles=[h1_legend, h2_legend, h3_legend, exact_legend])
plt.ylabel('y')
plt.xlabel('x')
plt.title('Euler\'s method')
plt.show()
