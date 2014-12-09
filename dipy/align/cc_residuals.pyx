cdef inline int _int_max(int a, int b) nogil:
    r"""
    Returns the maximum of a and b
    """
    return a if a >= b else b


cdef inline int _int_min(int a, int b) nogil:
    r"""
    Returns the minimum of a and b
    """
    return a if a <= b else b

cdef enum:
    SI = 0
    SI2 = 1
    SJ = 2
    SJ2 = 3
    SIJ = 4
    CNT = 5

def compute_cc_residuals(double[:,:,:] I, double[:,:,:] J, int radius):
    cdef:
        int ns = I.shape[0]
        int nr = I.shape[1]
        int nc = I.shape[2]
        double s1, s2, wx, p, t
        int i, j, k, s, r, c
        int start_k, end_k, start_i, end_i, start_j, end_j

        double[:,:,:] residuals = np.zeros((ns, nr, nc))

    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):

                    #Affine fit
                    s1 = 0
                    s2 = 0
                    wx = 0
                    p = 0
                    t = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                s1 += I[k, i, j]
                                s2 += I[k, i, j] * I[k, i, j]
                                wx += 1
                                p += I[k, i, j] * J[k, i, j]
                                t += J[k, i, j]

                    if s2 < 1e-9:
                        alpha = 0
                        beta = t/wx
                    else:
                        beta = (t - (s1 * p) / s2) / (wx - (s1 * s1) / s2)
                        alpha = (p - beta * s1) / s2

                    #Compute residuals
                    residuals[s, r, c] = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                residuals[s, r, c] += ((alpha * I[k, i, j] + beta) - J[k, i, j]) ** 2
    return residuals


def compute_cc_residuals_noboundary(double[:,:,:] I, double[:,:,:] J, int radius):
    cdef:
        int ns = I.shape[0]
        int nr = I.shape[1]
        int nc = I.shape[2]
        double s1, s2, wx, p, t, ave, worst
        int i, j, k, s, r, c, intersect
        int start_k, end_k, start_i, end_i, start_j, end_j

        double[:,:,:] residuals = np.zeros((ns, nr, nc))

    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):

                    #Affine fit
                    s1 = 0
                    s2 = 0
                    wx = 0
                    p = 0
                    t = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                intersect = (I[k, i, j]>0) * (J[k, i, j]>0)
                                if intersect == 0:
                                    continue
                                s1 += I[k, i, j]
                                s2 += I[k, i, j] * I[k, i, j]
                                wx += 1
                                p += I[k, i, j] * J[k, i, j]
                                t += J[k, i, j]

                    residuals[s, r, c] = 0
                    if wx<3:
                        continue
                    ave = t/wx

                    if s2 < 1e-6:
                        alpha = 0
                        beta = ave
                    else:
                        if s2 * wx - (s1 * s1) < 1e-6 and s2 * wx - (s1 * s1)  > -1e-6:
                            continue
                        beta = (t - (s1 * p) / s2) / (wx - (s1 * s1) / s2)
                        alpha = (p - beta * s1) / s2

                    #Compute residuals
                    worst = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                intersect = (I[k, i, j]>0) * (J[k, i, j]>0)
                                if intersect == 0:
                                    continue

                                worst += (ave - J[k, i, j]) ** 2
                                residuals[s, r, c] += ((alpha * I[k, i, j] + beta) - J[k, i, j]) ** 2
                    if residuals[s, r, c] > worst:
                        residuals[s, r, c] = worst

    return residuals