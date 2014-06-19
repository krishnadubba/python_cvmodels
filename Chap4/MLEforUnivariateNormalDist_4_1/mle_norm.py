#% Author: Stefan Stavrev 2013

#% Description: maximum likelihood learning of normal distribution.
#% Input: x - a row or a column vector.
#% Output: mu - mean, var - variance.
#function [mu, var] = mle_norm (x)
    #validate_input(x);
    
    #I = length(x);
    #mu = sum(x) / I;
    #var = sum((x - mu) .^ 2) / I;
#end

#% The input must be a row or a column vector.
#function [] = validate_input (x)
    #if ~(isrow(x) || iscolumn(x))
        #err = 'Invalid input: input must be a row or a column vector.';        
        #error(err);
    #end
#end
import sys
import numpy as np
from math import pow
    
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
    if (x.shape[0] == 1 or x.shape[1] == 1):
        return True
    else:
        err = 'Invalid input: input must be a row or a column vector.'
        print err
        return False