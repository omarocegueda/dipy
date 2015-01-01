import numpy as np
import scipy as sp
import scipy.linalg as linalg

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


def estimate_learning_rate(transform_type, dim, theta, step, domain_shape, domain_affine):
    h = 0.01
    max_step_size = 0.1 # maximum step length in physical units
    X = np.empty((2 ** dim, dim + 1)) # All 2^dim corners of the grid
    T0 = np.ndarray((dim + 1, dim + 1))
    T = np.ndarray((dim + 1, dim + 1))

    max_step = step.max()
    factor = h/max_step

    param_to_matrix(transform_type, dim, theta, T0)
    param_to_matrix(transform_type, dim, theta + factor * step, T)

    # Generate all corners of the given domain
    X[:, dim] = 1 # Homogeneous coordinate
    for i in range(2 ** dim):
        ii = i
        for j in range(dim):
            if (ii % 2) == 0:
                X[i, j] = 0
            else:
                X[i, j] = domain_shape[j] - 1
            ii = ii // 2
    X0 = X.copy()

    if domain_affine is not None:
        T0 = T0.dot(domain_affine)
        T = T.dot(domain_affine)

    X0 = X0.dot(T0.transpose())
    X = X.dot(T.transpose())

    sq_norms = np.sum((X - X0) ** 2, 1)
    step_scale = np.sqrt(sq_norms.max())/factor

    return max_step_size / step_scale


def golden_section_ls(val, theta, a, b, c):
    r""" Golden Section Line Search
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi

    tau = 0.2 #tolerance

    maxiter = 20
    niter = 0
    fb = val(theta * b)
    while niter < maxiter:
        if c - b > b - a:
            x = b + resphi * ( c - b )
        else:
            x = b - resphi * ( b - a )
        if np.abs(c - a) < tau * (np.abs(b) + np.abs(x))
            return (c + a) / 2.0

        fx = val(theta * x)
        if fx < fb:
            if c - b > b - a:
                a, b, c = b, x, c
            else:
                a, b, c = a, x, b
            fb = fx
        else:
            if c - b > b - a:
                a, b, c = a, b, x
            else:
                a, b, c = x, b, c
        niter += 1
    return (a + c) / 2.0


def nonlinear_cg(transform_type, dim, domain_shape, domain_affine,
                 val, grad, val_and_grad, x0, options):
    r""" Nonlinear Conjugate Gradient with Golden Section Line Search

    Parameters
    ----------
    val_and_grad : function
        a function receiving an array of parameters, and returns the value and
        gradient of the objective function
    x0 : array, shape(n,)
        startin point
    options : dictionary
        dictionary of optimization parameters
    """
    tol = options.tolerance
    theta = x0.copy()
    val0, grad0 = val_and_grad(theta)
    # The initial search direction is the steepest descent direction
    p = -1 * grad0
    while True:
        # Estimate maximum step size such that displacements are acceptably small
        rate = estimate_learning_rate(transform_type, dim, theta, p, domain_shape, domain_affine)
        # Minimize the objectove function along p
        alpha = golden_section_ls(val, p, 0, rate, 2 * rate)
        # Make a step along p with optimum length
        theta += alpha * p
        # Compute value and gradient at new current point
        val1, grad1 = val_and_grad(theta)
        # Compute the new (conjugate) search direction with Fletcher-Reeves rule
        beta = grad1.dot(grad1)/grad0.dot(grad0)
        p = beta * p - grad1
        # Prepare next iteration
        val0, grad0 = val1, grad1

    return theta

def test_conj_grad():
    A = np.array([[4, 1], [1, 3.0]])
    b = np.array([1,2.0])
    x0 = np.array([2,1.0])
    x0 = np.array([0,0.0])
    sol = conj_grad(A, b, x0)
    print("Sol:",sol)

if __name__ == '__main__':
    test_conj_grad()
