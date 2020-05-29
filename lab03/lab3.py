def f(x):
    return x**(1/2)

#Inputs : function, interval a to b and partition number N
def trapezoidal(func, a, b, N):
    delta_x = (b-a)/N
    sum = 0
    for i in range(1, N+1):
        p = a+((i-1)*delta_x) # Left endpoint
        q = a+(i*delta_x)     # Right endpoint
        sum = sum + (delta_x/2) * (f(p)+f(q)) #trap. formula
    print(f'trapezoidal: {sum}, partition number: {N}.')
    return sum

def simpson(func, a, b, N):
    delta_x = (b-a)/N
    sum = 0
    for i in range(1, N+1): #iterate 1 to N
        p = a+((i-1)*delta_x)
        q = a+(i*delta_x)
        mid = (p+q)/2         #midpoint of interval
        sum = sum + ((1/6)*delta_x)*(f(p)+ 4*f(mid)+f(q))
    print(f'Simpson: {sum}, partition number: {N}.')
    return sum


trapezoidal(f, 0, 2, 10)
trapezoidal(f, 0, 2, 100)
trapezoidal(f, 0, 2, 1000)
print("-------------------------------------------------")
simpson(f, 0, 2, 10)
simpson(f, 0, 2, 100)
simpson(f, 0, 2, 1000)

'''
According to WolframAlpha:
integral_0^2 sqrt(x) dx = (4 sqrt(2))/3≈1.88561808316413
And as we can see below the functions have a good
approximation of the integral. Especially as partition
numbers grow the approximation gets better. The error in
Simpson's rule is less than the trapezoid rule.

trapezoidal: 1.8682025382318126, partition number: 10.
trapezoidal: 1.8850418772248283, partition number: 100.
trapezoidal: 1.8855996071060295, partition number: 1000.
-------------------------------------------------
Simpson: 1.8830508367710803, partition number: 10.
Simpson: 1.885536898547035, partition number: 100.
Simpson: 1.8856155158810004, partition number: 1000.
'''
