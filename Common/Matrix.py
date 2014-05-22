% Author: Stefan Stavrev 2013


def isdiag(M):
    """ Description: return true if the matrix M is diagonal.
    """
    (m,n) = M.shape
    if m != n:
        return False
    for i in xrange(n):
        for j in xrange(n):
            if i != j and M[i,j] != 0:
                return False
    return True