# Q1
'''
⌈ 1    5 ⌉⌈ X_1 ⌉    ⌈ 7 ⌉
⌊-2   -7 ⌋⌊ X_2 ⌋ =  ⌊-5 ⌋
---------------------------------------
⌈ 1    5 |  7⌉ L_1
⌊-2   -7 | -5⌋ L_2
    L_2 = 2*L_1 + L_2 =>
⌈ 1    5 |  7⌉ L_1
⌊ 0    3 |  9⌋ L_2
    From L_2: 3*(x_2)=9 => x_2 = 3
    Set X_2 in L_1: (x_1) + 5*3 = 7 => x_1 = -8
⌈ X_1 ⌉   ⌈-8 ⌉
⌊ X_2 ⌋ = ⌊ 3 ⌋
'''

# Q2
'''
⌈ 0    2    1 ⌉⌈ X_1 ⌉   ⌈ -8 ⌉
| 1   -2   -3 || X_2 |   |  0 |
⌊-1    1    2 ⌋⌊ X_3 ⌋ = ⌊  3 ⌋
---------------------------------------
⌈ 0    2    1 | -8⌉ L_1
| 1   -2   -3 |  0| L_2
⌊-1    1    2 |  3⌋ L_3
    (L_1 <=> L_2) =>
⌈ 1   -2   -3 |  0⌉ L_2
| 0    2    1 | -8| L_1
⌊-1    1    2 |  3⌋ L_3
    L_3=L_2+L_3, =>
⌈ 1   -2   -3 |  0⌉ L_2
| 0    2    1 | -8| L_1
⌊ 0   -1   -1 |  3⌋ L_3
    L_3=(L_1/2)+L_3, =>
⌈ 1   -2   -3 |  0⌉ L_2
| 0    2    1 | -8| L_1
⌊ 0    0 -1/2 | -1⌋ L_3
    From L_3: (-1/2)*(x_3) = -1 => x_3 = 2
    Set X_3 in L_1: 2*(x_2) +2 = -8 => x_2 = -5
    Set both in L_2: 1*(X_1) -2*(-5) -3(2) = 0 =>
x_1 = -10+6 => x_1 = -4
⌈ X_1 ⌉   ⌈-4 ⌉
| X_1 |   |-5 |
⌊ X_2 ⌋ = ⌊ 2 ⌋
'''


# Q3
'''
⌈ 1   -2   -6 ⌉⌈ X_1 ⌉   ⌈ 12 ⌉
| 2    4   12 || X_2 |   |-17 |
⌊ 1   -4  -12 ⌋⌊ X_3 ⌋ = ⌊ 22 ⌋
---------------------------------------
⌈ 1   -2   -6 | 12⌉ L_1
| 2    4   12 |-17| L_2
⌊ 1   -4  -12 | 22⌋ L_3
    L_2=-2(L_1)+L_2,
    L_3=-1(L_1)+L_3 =>
⌈ 1   -2   -6 | 12⌉ L_1
| 0    8   24 |-41| L_2
⌊ 0   -2   -6 | 10⌋ L_3
    L_3=(1/4)(L_2)+L_3 =>
⌈ 1   -2   -6 | 12 ⌉ L_1
| 0    8   24 |-41 | L_2
⌊ 0    0    0 |-(1/4)⌋ L_3

As the last line L_3 gives 0=81/4 which is impossible
this system does not have a solution.
'''
# Q4
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



A=[[1,5],[-2,-7]]
b=[7,-5]
print(f'A={A} \nb={b} \nx={gauss(A,b)}\n{"-"*40}')
A = [[0,2,1],[1,-2,-3],[-1,1,2]]
b = [-8,0,3]
print(f'A={A} \nb={b} \nx={gauss(A,b)}\n{"-"*40}')
A = [[1,2,5,1],[3,-4,3,-2],[4,3,2,-1],[1,-2,-4,-1]]
b = [4,7,1,2]
print(f'A={A} \nb={b} \nx={gauss(A,b)}\n{"-"*40}')
