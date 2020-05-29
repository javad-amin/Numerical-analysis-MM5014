def qr(A):
    Q = calculateQ(getCols(A))
    R = calculateR(getCols(A), getCols(Q))
    return {"Q": Q,"R": R}

def calculateQ(A):
    Q= []
    for i in range(len(A)):
        v = A[i]
        for q in Q:
            v = vecSubt(v, scalarMul((dotProduct(q,A[i])), q))
        Q.append(scalarMul(1 / magnitude(v), v))
    return getCols(Q)


def calculateR(A, Q):
    n = len(A)
    R = [[0.0 for i in range(n)] for i in range(n)]
    V = [[0.0 for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i < j:
                R[i][j] = dotProduct(Q[i], A[j])
            V[j] = A[j]
            for k in range(j):
                V[j] = vecSubt(V[j], scalarMul(R[k][j], Q[k]))
            R[j][j] = magnitude(V[j])
            Q[j] = scalarMul(1/R[j][j], V[j])
    return R


def getCols(M):
    return [[M[i][j] for i in range(len(M)) for j in range(len(M[0]))][i::len(M)] for i in range(len(M))]


def magnitude(v):
    return (sum(x**2 for x in v))**(1/2)


def scalarMul(s,v2):
    return list(map(lambda x: s * x, v2))


def vecSubt(v1,v2):
    return [v1[i]-v2[i] for i in range(len(v1))]


def dotProduct(v1, v2):
    return sum([v1[i]*v2[i] for i in range(len(v1))])


def printMatrix(M):
    print("{frame}\n{m}\n{frame}\n".format(frame = "-"*len(str(roundMatrix(M)[0])), m = "\n".join([str(r) for r in roundMatrix(M)])))


def roundMatrix(M):
    return list(map(lambda x: list(map(lambda y: round(y,3),x)), M))



A = [[0, 1, 1],
    [1, 1, 2],
    [0, 0, 3]]

#
# A = [[3, 7, 36],
#     [31, 400, 1],
#     [2, 0, 8]]

print('Matrix A:')
printMatrix(A)
print(f'Q:')
printMatrix(qr(A)["Q"])
print(f'R:')
printMatrix(qr(A)["R"])
