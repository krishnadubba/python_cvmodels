from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

import os, sys
from scipy.stats import norm

dirname = os.path.dirname
cwd = dirname(os.path.realpath(__file__))
sys.path.append(dirname(dirname(cwd)))

from map_norm import map_norm
from Chap4.MLEforUnivariateNormalDist_4_1.mle_norm import mle_norm 

#% Generate random values from the normal distribution with
#% mean value original_mu and standard deviation original_sig.
original_mu  = 5
original_sig = 8
#% I can be modified in order to see how MAP behaves for small vs big
#% amounts of data.
I = 10000
r = np.random.normal(loc=original_mu, scale=original_sig, size=(I,1))

#% Estimate the mean and the variance for the data in r.
#% Values used for alpha, beta, gamma and delta are (1,1,1,0), for the
#% sake of the example. Other values can be tried too.
(estimated_mu, estimated_var) = map_norm(r, 1, 1, 1, 0)
estimated_sig                 = sqrt(estimated_var)

print 'Estimated mean: ' + repr(estimated_mu)

(mle_mu, mle_var) = mle_norm(r)
mle_sig           = sqrt(mle_var)

#% Estimate and print the error for the mean and the standard deviation.
muError  = abs(original_mu - estimated_mu)
sigError = abs(original_sig - estimated_sig)
print 'Errors: ' + repr(muError) + ', ' + repr(sigError)

#% Plot the original and the estimated models for comparison.
x = np.arange(-20,30,0.01)
original  = norm.pdf(x, loc=original_mu, scale=original_sig)
estimated = norm.pdf(x, loc=estimated_mu, scale=estimated_sig)
mle       = norm.pdf(x, loc=mle_mu, scale=mle_sig)

fig, ax = plt.subplots()
plt.xlabel('x',    fontsize=25)
plt.ylabel('P(x)', fontsize=25)
plt.title('MAP and MLE for Univariate Norm Dist: ' + repr(I) + ' data points', fontsize=25)
ax.plot(x, original, 'k',   label='Original')
ax.plot(x, estimated,'b--', label='MAP',linewidth=2.0)
ax.plot(x, mle,      'r:',  label='MLE',linewidth=2.0)

# Now add the legend with some customizations.
legend = ax.legend(loc='upper right', shadow=True)

frame  = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.show()
