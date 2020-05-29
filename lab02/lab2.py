def f(x):
    return x**4 - 5*x**3 + 9*x + 3


def g(x):
    return 2*x**2 + 5 - 2.71828182846**x


def derivative(f, a):
    h = 10**-10
    return (f(a + h) - f(a))/h


def newton(func, start, epsilon):
    initial_value = start
    i = 1
    while i <= 1000:
        root = start - func(start)/derivative(func, start)
        if abs((1-(start/root))) < epsilon:
            break
        i+=1
        start = root
    print(f'X0 = {initial_value}, root: {root}, Steps: {i}')
    return root



def secant(func, start, epsilon):
    initial_value = start
    next = start - 0.1
    k = 1
    while k <= 1000:
        if func(next) < epsilon:
            root = next
            break
        root = (start*func(next) - next*func(start))/(func(next) - func(start))
        if abs(next - start) < epsilon:
            break
        start = next
        next = root
        k+=1
    print(f'X0 = {initial_value}, root: {root}, Steps: {k}')
    return root

'''
Plotting the function in online applications like desmos we see that f cuts
the x axis at exactly 4.529 in the interval [4,6] and g cuts x axis at 3.276 in
interval [3,4].

The results obtained by running the below code show that in the newton method
for both functions when the initialization value is in the specified range we
find a result relatively close to the actual root.  And as the initial guess is
selected to be some value closer to the root the steps which it takes in our
iteration to come to the result are less.

As a drawback of the newton method we can see that sometimes we might not get
the correct root, for example when the slope of the tanget line is 0 meaning our first derivitive is 0. Also if the initialization value is far away from the root it takes more steps to the to the result in case of function f.

Where as in secant method works even when the initial guess is at a point with slope 0 as the second point in our second line will make sure that the second line wont have slope of 0.

One observation is that the newton method seems to generate results closer to
the root we are looking for in the specified interval where as secant sometimes
give results which are not optimised like when x0 = 4 in function f. Same result
can be seen more obviously in function g for x0=3, 4, ... where newton method
gives a better result for the same guesses.

Performance-wise we observe that the steps which it takes to get to the result
in the secant method seems to less than the ones in newton method when the
x0 we choose is close to the root, sometimes getting the result in one step
but as our guess gets further away from the root then the steps in secant are
more.
'''

print(f'\nNewton - f:')
newton(f, 4, 0.001)
newton(f, 4.1, 0.001)
newton(f, 4.3, 0.001)
newton(f, 4.4, 0.001)
newton(f, 4.5, 0.001)
newton(f, 4.7, 0.001)
newton(f, 5, 0.001)
newton(f, 5.6, 0.001)
newton(f, 6, 0.001)
newton(f, 300, 0.001)


print(f'\nSecant - f')
secant(f, 4, 0.001)
secant(f, 4.1, 0.001)
secant(f, 4.3, 0.001)
secant(f, 4.4, 0.001)
secant(f, 4.5, 0.001)
secant(f, 4.7, 0.001)
secant(f, 5, 0.001)
secant(f, 5.6, 0.001)
secant(f, 6, 0.001)
secant(f, 300, 0.001)

print(f'\nNewton - g:')
newton(g, 3, 0.001)
newton(g, 3.1, 0.001)
newton(g, 3.3, 0.001)
newton(g, 3.4, 0.001)
newton(g, 3.5, 0.001)
newton(g, 3.7, 0.001)
newton(g, 3.8, 0.001)
newton(g, 3.9, 0.001)
newton(g, 4, 0.001)

print(f'\nSecant - g')
secant(g, 3, 0.001)
secant(g, 3.1, 0.001)
secant(g, 3.3, 0.001)
secant(g, 3.4, 0.001)
secant(g, 3.5, 0.001)
secant(g, 3.7, 0.001)
secant(g, 3.8, 0.001)
secant(g, 3.9, 0.001)
secant(g, 4, 0.001)
