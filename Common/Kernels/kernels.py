def kernel_linear(x_i, x_j)
    """ Linear kernel function.
        Input: x_i - a column vector
               x_j - a column vector
    """
    f = x_i' * x_j;
    return f
