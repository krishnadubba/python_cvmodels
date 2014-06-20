# Description: maximum likelihood learning of normal distribution.
# Input: x - a row or a column vector.
# Output: mu - mean, var - variance.
import sys
import numpy as np
    
def mle_norm(x):
    
    if not validate_mle_input(x):
        sys.exit()

    I = len(x);
    mu = sum(x) / I
    var = np.sum(np.power(x - mu, 2)) / I
    return (mu, var)

# The input x must be a row or a column vector.
# The parameters alpha, beta, gamma must be strictly greater than zero.
def validate_mle_input(x):
    if (len(x.shape) == 1 or x.shape[0] == 1 or x.shape[1] == 1):
        return True
    else:
        print 'Invalid input: input must be a row or a column vector.'
        return False