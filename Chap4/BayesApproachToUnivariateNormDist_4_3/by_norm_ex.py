# Generate random values from the normal distribution with
# mean value original_mu and standard deviation original_sig.

from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

import os, sys
from scipy.stats import norm

from by_norm import by_norm

dirname = os.path.dirname
cwd = dirname(os.path.realpath(__file__))
sys.path.append(dirname(dirname(cwd)))

from Chap4.MAPforUnivariateNormalDist_4_2.map_norm import map_norm 

original_mu  = 5
original_sig = 8
I = 5
r = np.random.normal(loc=original_mu, scale=original_sig, size=(I,1))

# Estimate the mean and the variance for the data in r.
# Values used for alpha, beta, gamma and delta are (1,1,1,0), for the
# sake of the example. Other values can be tried too.
x_test = np.arange(-20,30,0.01)
[alpha_post, beta_post, gamma_post, delta_post, x_prediction] = by_norm(r, 1, 1, 1, 0, x_test)

# MAP, for comparison purposes to the Bayesian approach.
[map_mu, map_var] = map_norm(r, 1, 1, 1, 0)
map_sig = sqrt(map_var)

# Plot the original and the estimated models for comparison.
original      = norm.pdf(x_test, loc=original_mu, scale=original_sig)
map_estimated = norm.pdf(x_test, loc=map_mu, scale=map_sig)

fig, ax = plt.subplots()
plt.xlabel('x',    fontsize=25)
plt.ylabel('P(x)', fontsize=25)
plt.title('MAP and Bayesian for Univariate Norm Dist: ' + repr(I) + ' data points', fontsize=25)
ax.plot(x_test, original,     'k',   label='Original')
ax.plot(x_test, map_estimated,'b--', label='MAP',linewidth=2.0)
ax.plot(x_test, x_prediction, 'r:',  label='Bayesian',linewidth=2.0)

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
