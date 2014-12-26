import numpy as np
def conj_grad(A, b, x0):
    tol = 1e-12

    xnew = x0
    rold = b - A.dot(xnew)
    pnew = rold

    sqerr = rold.dot(rold)
    while tol < sqerr:
        alpha = pnew.dot(rold) / pnew.dot(A).dot(pnew)
        xold = xnew
        xnew = xold + alpha * pnew

        rnew = b - A.dot(xnew)
        beta = rnew.dot(rnew) / rold.dot(rold)
        rold = rnew

        pnew = rnew + beta * pnew
        sqerr = rold.dot(rold)

    return xnew


def test_conj_grad():
    A = np.array([[4, 1], [1, 3.0]])
    b = np.array([1,2.0])
    x0 = np.array([2,1.0])
    x0 = np.array([0,0.0])
    sol = conj_grad(A, b, x0)
    print("Sol:",sol)

if __name__ == '__main__':
    test_conj_grad()
