# Generate random values from the normal distribution with
# mean value original_mu and standard deviation original_sig.

import numpy as  np
import os, sys
import matplotlib.pyplot as plt

from math import sqrt
from scipy.stats import norm

dirname = os.path.dirname
cwd = dirname(os.path.realpath(__file__))
sys.path.append(dirname(dirname(cwd)))

from mle_norm import mle_norm

original_mu  = 5
original_sig = 8

# Change size to see how MLE fits
I = 100
r = np.random.normal(loc=original_mu, scale=original_sig, size=(I,1))

# Estimate the mean and the variance for the data in r.
[estimated_mu, estimated_var] = mle_norm(r)
estimated_sig = sqrt(estimated_var)

# Estimate and print the error for the mean and the standard deviation.
muError  = abs(original_mu - estimated_mu)
sigError = abs(original_sig - estimated_sig)
print muError, sigError

# Plot the original and the estimated models for comparison.
x = np.arange(-20,30,0.01)
original  = norm.pdf(x, loc=original_mu, scale=original_sig)
estimated = norm.pdf(x, loc=estimated_mu, scale=estimated_sig)

fig, ax = plt.subplots()
plt.xlabel('x',    fontsize=25)
plt.ylabel('P(x)', fontsize=25)
plt.title('MLE for Univariate Normal Distribution', fontsize=25)
ax.plot(x, original, 'k', label='Original')
ax.plot(x, estimated, 'k--', label='Estimated')

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
