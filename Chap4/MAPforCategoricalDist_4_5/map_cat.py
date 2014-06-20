# Description: MAP learning of categorical distribution.
# Input: x - a row or a column vector,
#        alpha - Dirichlet prior.
# Output: theta - K categorical distribution parameters.
import sys
import numpy as np

def map_cat(x, alpha):
    if not validate_cat_input(x, K):
        sys.exit()    
    
    I = len(x)
    K = len(alpha)
    counts = np.histogram(x, K)
    theta  = np.divide((counts - 1 + alpha), (I - K + sum(alpha)))
    return theta    
    
# The input x must be a row or a column vector  .
def validata_input(x):
    if (len(x.shape) == 1 or x.shape[0] == 1 or x.shape[1] == 1):
        return True
    else:
        print 'Invalid input: input must be a row or a column vector.'
        return False
    