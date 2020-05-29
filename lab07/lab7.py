# Q1
'''
⌈ 5   -2    3 ⌉⌈ X_1 ⌉   ⌈ -1 ⌉
|-3    9    1 || X_2 |   |  2 |
⌊ 2   -1   -7 ⌋⌊ X_3 ⌋ = ⌊  3 ⌋
---------------------------------------
⌈ 5   -2    3 | -1⌉ L_1
|-3    9    1 |  2| L_2
⌊ 2   -1   -7 |  3⌋ L_3
    L_1 = L_1/5 =>
⌈ 1 -2/5  3/5 | -1/5⌉ L_1
|-3    9    1 |    2| L_2
⌊ 2   -1   -7 |    3⌋ L_3
    L_2 = L_2+(3*L_1),
    L_3 = L_3+(-2*L_1) =>
⌈ 1  -2/5   3/5 | -1/5⌉ L_1
| 0  39/5  14/5 |  7/5| L_2
⌊ 0  -1/5 -41/5 | 17/5⌋ L_3
    L_2 = L2*(5/39),
    L_3 = L3*(-5) =>
⌈ 1  -2/5    3/5 | -1/5 ⌉ L_1
| 0     1  14/39 |  7/39| L_2
⌊ 0     1     41 | -17  ⌋ L_3
    L_3 = L3+(-L_2) =>
⌈ 1  -2/5     3/5   |   -1/5   ⌉ L_1
| 0     1    14/39  |    7/39  | L_2
⌊ 0     0  1585/39  | -670/39  ⌋ L_3
    L_3 = L3*(39/1585) =>
⌈ 1  -2/5     3/5   | -1/5     ⌉ L_1
| 0     1    14/39  |  7/39    | L_2
⌊ 0     0    1      | -134/317 ⌋ L_3
    L_2 = L2+(-14/39 * L_3),
    L_1 = L1+(-3/5 * L_3) =>
⌈ 1  -2/5    0  |   17/317 ⌉ L_1
| 0     1    0  |  105/317 | L_2
⌊ 0     0    1  | -134/317 ⌋ L_3
    L_1 = L1+(2/5 * L_2) =>
⌈ 1   0    0  |   59/317 ⌉ L_1
| 0   1    0  |  105/317 | L_2
⌊ 0   0    1  | -134/317 ⌋ L_3


⌈ X_1 ⌉   ⌈   59/317 ⌉   ⌈  0,186119874 ⌉
| X_2 |   |  105/317 |   |  0,331230284 |
⌊ X_3 ⌋ = ⌊ -134/317 ⌋ = ⌊ -0,422712934 ⌋
'''
#Q2
def jacobi(A,b):
    dInverse = getDInverse(A)
    LPlusU = matAdd(getL(A), getU(A))
    x = [0 for i in range(len(b))]
    for i in range(500):
        x_k = dot(dInverse, vecSubt(b, dot(LPlusU, x)))
        if magnitude(vecSubt(x_k, x)) < 0.00000000001:
            return x_k
        x = x_k
    return x


def getDInverse(A):
    return [[1 / A[i][j] if i == j else 0 for j in range(len(A))] for i in range(len(A))]


def getU(A):
    return [[A[i][j] if i < j else 0 for j in range(len(A))] for i in range(len(A))]


def getL(A):
    return [[A[i][j] if i > j else 0 for j in range(len(A))] for i in range(len(A))]


def dot(m, v):
    return [sum(x*y for x,y in zip(vm,v)) for vm in m]


def matAdd(m1, m2):
    return [list(x+y for x,y in zip(v1,v2)) for v1,v2 in zip(m1,m2)]


def vecSubt(v1, v2):
    return list(x-y for x,y in zip(v1,v2))


def magnitude(v):
    return (sum(x**2 for x in v))**(1/2)


def printMatrix(M):
    print("{frame}\n{m}\n{frame}\n".format(frame = "-"*len(str(roundMatrix(M)[0])), m = "\n".join([str(r) for r in roundMatrix(M)])))


def roundMatrix(M):
    return list(map(lambda x: list(map(lambda y: round(y,3),x)), M))

#Q3. compared it's fine
A = [[5,-2,3],
    [-3,9,1],
    [2,-1,-7]]
b = [-1,2,3]

print('A: ')
printMatrix(A)
print(f'b: {b}\n')
print(f'Jacobi method x: \n{[round(x, 9) for x in jacobi(A, b)]}\n')


#Q4
def gradientDescent(f,gradient,epsilon,M,lSize):
    x = [-1.4,2] # Would be better if it was input
    for i in range(M):
        old_x = x
        x = vecSubt(old_x, scalarMul(lSize, gradient(old_x)))

        step = vecSubt(x, old_x)
        if magnitude(step) <= epsilon:
            break
    return (round(f(x), 2), [round(x,2) for x in x])



def scalarMul(s,v2):
    return list(map(lambda x: s * x, v2))



a = 1 # Difficulity
b = 5 # Difficulity


def f(x):
    return (a-x[0])**2+b*(x[1]-x[0]**2)**2


def gradient(x):
    return [-4*b*x[0]*(x[1]-x[0]**2)-2*(a-x[0]), 2*b*(x[1]-x[0]**2)]

print('Gradient descent:')
print(gradientDescent(f, gradient, 10**-100, 100000000, 0.01))
