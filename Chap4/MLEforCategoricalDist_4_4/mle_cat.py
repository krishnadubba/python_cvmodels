# Description: maximum likelihood learning of categorical distribution.
# Input: x - a row or a column vector,
#        K - number of categories.
# Output: theta - K categorical distribution parameters.
import sys
import numpy as np

def mle_cat(x, K):
    if not validate_cat_input(x, K):
        sys.exit()    
    
    # [counts] = hist(x, 1:K);
    # theta = counts ./ sum(counts);    
    counts = np.histogram(x, K)
    theta  = np.divide(counts, sum(counts))
    return theta    
    
# The input x must be a row or a column vector, K must be scalar.
def validata_cat_input(x, K):
    if (len(x.shape) == 1 or x.shape[0] == 1 or x.shape[1] == 1):
        pass
    else:
        print 'Invalid input: input must be a row or a column vector.'
        return False
    if not np.isscalar(K):
        print 'Invalid input: K must be a scalar value.'
        return False
    return True