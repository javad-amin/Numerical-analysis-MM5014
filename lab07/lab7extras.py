#Cool functions not used but I keep them

def getCols(M):
    return [[M[i][j] for i in range(len(M)) for j in range(len(M[0]))][i::len(M)] for i in range(len(M))]


def matDot(m1, m2):
    return [[vecDot(v1, v2) for v1 in m1 for v2 in getCols(m2)][i:i+len(m1)] for i in range(0,len(m1)**2,len(m1))]


def matVecDot(m, v):
    return [vecDot(vm, v) for vm in m]


def vecDot(v1, v2):
    return sum(x*y for x,y in zip(v1,v2))


def matAdd(m1, m2):
    return [vecAdd(v1, v2) for v1,v2 in zip(m1,m2)]


def vecAdd(v1, v2):
    return list(x+y for x,y in zip(v1,v2))


def vecSubt(v1, v2):
    return list(x-y for x,y in zip(v1,v2))


def magnitude(v):
    return (sum(x**2 for x in v))**(1/2)
