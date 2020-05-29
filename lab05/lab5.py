def gauss(A, b):
    appendVectorToMatrix(A, b)
    upperTriagularMatrix(A)
    return findX(A)


def appendVectorToMatrix(A, b):
    for i in range(len(A)):
        A[i].append(b[i])


def upperTriagularMatrix(A):
    n, r =len(A), len(A[0])
    for k in range(n):
        pivoting(A, k)
        printMatrix(A)
        for i in range(k+1, n):
            m = A[i][k]/A[k][k]
            for j in range(r):
                A[i][j] -= m*A[k][j]
        print(f'Reducing row items under a_{k+1}_{k+1} to 0:')
        oneInDiagonals(A, k, A[k][k]) #Making sure the diagonals are 1
        printMatrix(A)


def pivoting(A, k):
    max, mRow = abs(A[k][k]), k
    for i in range(k+1, len(A)):
        next = abs(A[i][k])
        if next > max:
            max = next
            mRow = i
    swap(A, mRow, k)

def swap(A, r1, r2):
    if r1 != r2:
        temp, A[r1] = A[r1], A[r2]
        A[r2] = temp
        print(f'<<< Row {r1 + 1} was swapped with row {r2 + 1} >>>')
    else:
        print(f'<<< No row swapping >>>')

def oneInDiagonals(A, r, diagonal):
    for i in range(len(A[r])):
        A[r][i] = A[r][i] / diagonal

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

def printMatrix(M):
    print('-'*len(str(M[0])))
    for r in M:
        for i in range(len(r)):
            r[i] = round(r[i], 3)
        print(r)
    print('-'*len(str(M[0])))
    print('')

A = [[0.143, 0.357, 2.01],
    [-1.31, 0.911, 1.99],
    [11.2, -4.30, -0.605]]
b = [-5.173, -5.458, 4.415]
print('Matrix A:')
printMatrix(A)
print(f'b = {b}\n')
result = gauss(A,b)
for r in range(len(result)):
    print(f'x_{r+1} = {round(result[r], 3)}')
