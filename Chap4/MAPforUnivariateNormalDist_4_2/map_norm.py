# Description: maximum a posteriori learning of normal distribution.
# Input: x - a row or a column vector, (alpha, beta, gamma, delta) - 
# parameters for the conjugate prior.
# Output: mu - mean, var - variance.
#function [mu, var] = map_norm (x, alpha, beta, gamma, delta)
    #validate_input(x, alpha, beta, gamma);

    #I = length(x);
    #mu = (sum(x) + gamma*delta) / (I + gamma);
    #var_up = sum((x - mu) .^ 2) + 2*beta + gamma*(delta-mu)^2;
    #var_down = I + 3 + 2*alpha;
    #var = var_up / var_down;
#end

def map_norm(x, alpha, beta, gamma, delta):
    import sys
    import numpy as np
    from math import pow
    
    if not validate_map_input(x, alpha, beta, gamma):
        sys.exit()

    I = len(x);
    mu = (sum(x) + gamma*delta) / (I + gamma)
    var_up = np.sum(np.power(x - mu, 2)) + 2*beta + gamma*pow((delta-mu),2)
    var_down = I + 3 + 2*alpha
    var = var_up / var_down
    return (mu, var)

# The input x must be a row or a column vector.
# The parameters alpha, beta, gamma must be strictly greater than zero.
def validate_map_input(x, alpha, beta, gamma):
    if (x.shape[0] == 1 or x.shape[1] == 1):
        pass
    else:
        err = 'Invalid input: input must be a row or a column vector.'
        print err
        return False
    
    if alpha <= 0 or beta <= 0 or gamma <= 0:
        a = 'Invalid prior parameters: alpha, beta and gamma must be';
        b = ' strictly greater than zero.';
        print a, b
        return False
    
    return True