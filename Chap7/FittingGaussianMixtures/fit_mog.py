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
from math import log
import scipy.stats.multivariate_normal as mvn

def fit_mog (x, K, precision):
    weights = [1.0/K] * K
    
    # Initialize the values in mu to K randomly chosen unique datapoints.
    I = x.shape[0]  # Note array index in python starts from 0
    K_random_unique_ints = np.random.sample(I, K)
    mu = x[K_random_unique_ints,:]
    
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
        
        # Compute the numerator of Bayes' rule.
        for k in xrange(K):
            l[:,k] = weights[k] * mvn.pdf(x, mu[k,:], sig[k])
        
        # Compute the responsibilities by normalizing.    
        s = sum(l,2);        
        for i in xrange(I):
            r[i,:] = np.divide(l[i,:], s[i])
         
        #  Maximization step.
        r_summed_rows = sum(r,1)
        r_summed_all  = sum(sum(r,1),2)
        for k in xrange(K):
            # Update lambda.
            weights[k] = r_summed_rows(k) / r_summed_all
            
            # Update mu
            new_mu = np.zeros(1,dimensionality)
            for i in xrange(I):
                new_mu = new_mu + r[i,k]*x[i,:]
            mu[k,:] = np.divide(new_mu, r_summed_rows[k])
            
            # Update sigma
            new_sigma = np.zeros(dimensionality,dimensionality)
            for i in xrange(I):
                mat = x[i,:] - mu[k,:]
                mat = r[i,k] * np.multiply(mat.T, mat)
                new_sigma = new_sigma + mat
            sig[k] = np.divide(new_sigma, r_summed_rows[k])
            
        # Compute the log likelihood L.
        temp = np.zeros(I,K)
        for k in xrange(K):
            temp[:,k] = weights[k] * mvn.pdf(x, mu[k,:], sig[k])
            
        temp = sum(temp,2)
        temp = log(temp)       
        L = sum(temp)
        
        iterations = iterations + 1;        
        if abs(L - previous_L) < precision:
            break
        
        previous_L = L
        
    return (weights, mu, sig)

