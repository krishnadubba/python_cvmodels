# Description: Fitting mixture of Gaussians.
# Input: x       - each row is one datapoint.
#        K       - number of Gaussians in the mixture.
#        precision - the algorithm stops when the difference between
#                    the previous and the new likelihood is < precision.
#                    Typically this is a small number like 0.01.
# Output:
#        lambda  - lambda(k) is the weight for the k-th Gaussian.
#        mu      - mu(k,:) is the mean for the k-th Gaussian.
#        sig     - sig{k} is the covariance matrix for the k-th Gaussian.

import numpy as np

def fit_mog (x, K, precision):
    weights = [1.0/K] * K
    
    # Initialize the values in mu to K randomly chosen unique datapoints.
    # randomly select K data points from x 
    I = x.shape[0]  # Note array index in python starts from 0
    K_random_unique_integers = randperm(I);
    K_random_unique_integers = K_random_unique_integers(1:K);
    mu = x (K_random_unique_integers,:)
    
    sig = dict()
    dimensionality   = x.shape[1]
    dataset_mean     = np.divide(sum(x,1), I)
    dataset_variance = np.zeros(dimensionality, dimensionality)
    for i in xrange(I):
        mat = x[i,:] - dataset_mean
        mat = np.multiply(mat.T, mat)
        dataset_variance = dataset_variance + mat
    dataset_variance = np.divide(dataset_variance, I)
    for i in xrange(K):
        sig[i] = dataset_variance
    
    # The main loop.
    iterations = 0    
    previous_L = 1000000 # just a random initialization
    while true:
        # Expectation step.
        l = np.zeros(I,K)
        r = np.zeros(I,K)
        for k in xrange(K):
            l[:,k] = weights(k) * mvnpdf(x, mu[k,:], sig[k])
            
        s = sum(l,2);        
        for i in xrange(I):
            r[i,:] = np.divide(l[i,:], s[i])
    
        

    return [weights, mu, sig]

