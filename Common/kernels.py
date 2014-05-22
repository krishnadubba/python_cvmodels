def kernel_linear(x_i, x_j):
    """ Linear kernel function.
        Input: x_i - a numpy column vector
               x_j - a numpy column vector
    """
    f = x_i.transpose() * x_j;
    return f

def kernel_gauss(x_i, x_j, lambd):
    """ Gaussian kernel function.
        Input: x_i - a numpy column vector
               x_j - a numpy column vector    
    """
    import math
    
    x_diff = x_i - x_j    
    temp = x_diff.transpose() * x_diff
    f = math.exp(-0.5*temp/pow(lambd,2))
    return f

if __name__ == '__main__':
    import numpy as np
    a = np.matrix([1,2,3]).transpose()
    b = np.matrix([4,5,6]).transpose()
    print kernel_linear(a,b)
    print kernel_gauss(a,b,2)