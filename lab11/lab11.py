import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import exp, sqrt

# lab4 code used for guessian elimination dont mind it scroll down
def gauss(A, b):
    appendVectorToMatrix(A, b)
    firstRowFirstItemNotZero(A)
    upperTriagularMatrix(A)
    return findX(A)


def appendVectorToMatrix(A, b):
    for i in range(len(A)):
        A[i].append(b[i])


def firstRowFirstItemNotZero(A):
    if A[0][0] == 0:
        for i in range(len(A)):
            if A[i][0] != 0:
                temp, A[i] = A[i], A[0]
                A[0] = temp
                return A
        raise ValueError('All rows start by 0.')
    return A


def upperTriagularMatrix(A):
    n, r =len(A), len(A[0])
    for k in range(n):
        for i in range(k+1, n):
            m = A[i][k]/A[k][k]
            for j in range(r):
                A[i][j] -= m*A[k][j]

def findX(A):
    n = len(A)
    i = n - 1
    x = [0 for i in range(n)]
    for e in A[::-1]: #iterating backwards
            x[i] = e[n] / e[i] #b_i devided by diagonal
            k = i
            while k >= 0:
                A[k][n] -= A[k][i] * x[i] # setting x_i in lines above
                k -= 1
            i -= 1
    return x



'''
###   New code for lab 11    ###
### Finite difference method ###
'''
def finDifMethod(p,q,r,a,b,x0,xn,h):
    n = int((b-a)/h)
    A = [[0 for i in range(n-1)] for i in range(n-1)]
    m = len(A)
    # Filling in A
    for i in range(m):
        if i != 0:
            A[i][i-1] = 1+h/2*p
        A[i][i] = -2-h**2*q
        if i != m-1:
            A[i][i+1] = 1-h/2*p
    f = [0 for i in range(m)]
    # Creating vector f
    for i in range(m):
        if i == 0:
            f[i] = h**2*r-(1+h/2*p)*x0
        else:
            f[i] = h**2*r
        if i == m-1:
            f[i] = h**2*r-(1-h/2*p)*xn
    res = gauss(A,f)
    res.insert(0,x0)
    res.append(xn)
    return res


def blackBox(f,a,b,x0,z,h):
    x1 = x0 + z*h
    res = [x0]
    while round(a, 2) <= b-h:
        newX = f(x0,x1,h)
        res.append(newX)
        x0 = res[-2]
        x1 = res[-1]
        a += h
    return res

def secant(yz1, yz2, z1, z2, beta):
    return z2 - (yz2-beta) * (z2-z1)/((yz2-beta)-(yz1-beta))

def shootMethod(f,yz,a,b,x0,xn,z1,z2,h):
    yz1 = yz(f,a,b,x0,z1,h)[-1]
    yz2 = yz(f,a,b,x0,z2,h)[-1]
    newZ = secant(yz1, yz2, z1, z2, xn)
    return newZ

def f(x0,x1,h):
    return 2*x1-x0+(3*h**2*x1)/2

def exactX(t):
    return (exp(-sqrt(3/2)*t)*(-4*exp(sqrt(6)*t)+exp(sqrt(3/2)*(2*t+1))-exp(sqrt(3/2))+4*exp(sqrt(6))))/(exp(sqrt(6))-1)

def exactXValues(f,start, end, h):
    res = []
    while round(start, 2) <= end:
        res.append(f(start))
        start += h
    return res

def tRange(t, step, end):
    res = []
    while round(t, 2) <= end:
        res.append(round(t, 2))
        t += step
    return res

p, q, r = 0, 3/2, 0
t0, tn = 0, 1
x0, xn = 4, 1
h1, h2 = 0.1, 0.01

# x''(t) = (3/2)*x(t)
# y'' = p(x)y' + q(x)y + r(x)

tValues = tRange(t0, h1, tn)
xValues = finDifMethod(p,q,r,t0,tn,x0,xn,h1)

t2Values = tRange(t0, h2, tn)
x2Values = finDifMethod(p,q,r,t0,tn,x0,xn,h2)
xValuesExact = exactXValues(exactX,t0,tn,h2)


z1 = -2
z2 = -3
zStar = shootMethod(f,blackBox,t0,tn,x0,xn,z1,z2,h1)
xValuesZ1Shooting = blackBox(f,t0,tn,x0,z1,h1)
xValuesZ2Shooting = blackBox(f,t0,tn,x0,z2,h1)
xValuesZStarShooting = blackBox(f,t0,tn,x0,zStar,h1)


plt.subplot(221)
h1Col, h2Col, exactCol = "red", "blue", "green"
plt.plot(tValues, xValues, color = h1Col)
h1_legend = mpatches.Patch(color=h1Col, label=str(h1))
plt.plot(t2Values, x2Values, color=h2Col)
h2_legend = mpatches.Patch(color=h2Col, label=str(h2))
plt.plot(t2Values, xValuesExact, color = exactCol)
exact_legend = mpatches.Patch(color=exactCol, label="exact")
plt.legend(handles=[h1_legend, h2_legend, exact_legend])
plt.ylabel('y')
plt.xlabel('x')
plt.title("Finite difference method")


plt.subplot(223)
z1Col, z2Col, zStarCol = "red", "blue", "green"
plt.plot(tValues, xValuesZ1Shooting, color = z1Col)
z1_legend = mpatches.Patch(color=z1Col, label=str(z1))
plt.plot(tValues, xValuesZ2Shooting, color=z2Col)
z2_legend = mpatches.Patch(color=h2Col, label=str(z2))
plt.plot(tValues, xValuesZStarShooting, color = zStarCol)
zStar_legend = mpatches.Patch(color=zStarCol, label=f'zStar: {zStar}')
plt.legend(handles=[z1_legend, z2_legend, zStar_legend])
plt.ylabel('y')
plt.xlabel('x')
plt.title("Shooting method h = 0.1")


z1 = -2
z2 = -3
zStar = shootMethod(f,blackBox,t0,tn,x0,xn,z1,z2,h2)
xValuesZ1Shooting = blackBox(f,t0,tn,x0,z1,h2)
xValuesZ2Shooting = blackBox(f,t0,tn,x0,z2,h2)
xValuesZStarShooting = blackBox(f,t0,tn,x0,zStar,h2)


plt.subplot(224)
z1Col, z2Col, zStarCol = "red", "blue", "green"
plt.plot(t2Values, xValuesZ1Shooting, color = z1Col)
z1_legend = mpatches.Patch(color=z1Col, label=str(z1))
plt.plot(t2Values, xValuesZ2Shooting, color=z2Col)
z2_legend = mpatches.Patch(color=h2Col, label=str(z2))
plt.plot(t2Values, xValuesZStarShooting, color = zStarCol)
zStar_legend = mpatches.Patch(color=zStarCol, label=f'zStar: {zStar}')
plt.legend(handles=[z1_legend, z2_legend, zStar_legend])
plt.ylabel('y')
plt.xlabel('x')
plt.title("Shooting method h = 0.01")

plt.show()
