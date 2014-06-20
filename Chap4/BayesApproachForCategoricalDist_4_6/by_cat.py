# Description: Bayesian approach to categorical distribution.
# Input: x - a row or a column vector,
#        alpha_prior - prior parameters.
# Output: alpha_post - posterior parameters,
#         prediction - predictions for all categories.

import numpy as np

def by_cat(x, alpha_prior):
    if not validate_input(x):
        sys.exit()

    # Compute posterior.
    K          = len(alpha_prior)
    counts     = np.histogram(x, K)
    alpha_post = alpha_prior + counts
        
    # Predict.
    prediction = np.divide(alpha_post, sum(alpha_post))
    
    return (alpha_post, prediction)
    
# The input x must be a row or a column vector  .
def validata_input(x):
    if (len(x.shape) == 1 or x.shape[0] == 1 or x.shape[1] == 1):
        return True
    else:
        print 'Invalid input: input must be a row or a column vector.'
        return False