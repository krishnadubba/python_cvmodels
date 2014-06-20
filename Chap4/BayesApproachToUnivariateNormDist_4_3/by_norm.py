# Description: Bayesian approach to univariate normal distribution.
# Input: x - a row or a column vector, prior parameters, and test data.
# Output: posterior parameters, and prediction values for the test data.
def by_norm(x, alpha_prior, beta_prior, gamma_prior, delta_prior, x_test):
    import sys
    import numpy as np
    from math import sqrt
    from scipy.stats.distributions import gamma as gamma_dist
    
    if not validate_bynorm_input(x, alpha_prior, beta_prior, gamma_prior, x_test):
        sys.exit()
    
    # Compute posterior parameters.
    I = len(x)
    alpha_post = alpha_prior + I/2
    beta_post = np.sum(np.power(x,2))/2 + beta_prior + (gamma_prior*pow(delta_prior,2))/2 \
        - pow((gamma_prior*delta_prior + sum(x)),2) / (2*(gamma_prior + I))
    gamma_post = gamma_prior + I
    delta_post = (gamma_prior*delta_prior + sum(x)) / (gamma_prior + I)
    
    # Compute intermediate parameters.
    alpha_int = alpha_post + 0.5
    beta_int = np.power(x_test,2)/2 + beta_post + pow(gamma_post*delta_post,2)/2 \
        - np.power(gamma_post*delta_post + x_test, 2) / (2*gamma_post + 2)
    gamma_int = gamma_post + 1
        
    # Predict values for x_test.
    temp1 = sqrt(gamma_post) * pow(beta_post, alpha_post) * gamma_dist(alpha_int)
    x_prediction_up = np.repeat(temp1, 1, len(x_test))
    x_prediction_down = sqrt(2*np.pi) * sqrt(gamma_int) * gamma_dist(alpha_post) \
        * np.power(beta_int,alpha_int)
    x_prediction =  np.divide(x_prediction_up, x_prediction_down)
    
    return (alpha_post, beta_post, gamma_post, delta_post, x_prediction)
    
# The inputs x and x_test must be row or column vectors.
# The parameters alpha, beta, gamma must be strictly greater than zero.
def validate_bynorm_input(x, alpha, beta, gamma, x_test):
    if (len(x.shape) == 1 or x.shape[0] == 1 or x.shape[1] == 1):
        pass
    else:
        err = 'Invalid input: input must be a row or a column vector.'
        print err
        return False

    if (len(x_test.shape) == 1 or x_test.shape[0] == 1 or x_test.shape[1] == 1):
        pass
    else:
        err = 'Invalid input: input must be a row or a column vector.'
        print err
        return False
    
    if alpha <= 0 or beta <= 0 or gamma <= 0:
        print 'Invalid prior parameters: alpha, beta and gamma must be strictly greater than zero.'
        return False
    
    return True
