import matplotlib.pyplot as plt
import numpy as np

from mle_cat import mle_cat

# Generate random values from the categorical distribution 
# with 6 categories and the corresponding probabilities.
original_probabilities = [0.25, 0.15, 0.1, 0.1, 0.15, 0.25];
r1 = np.random.choice(6, size=200,   replace=True, p=original_probabilities)
r2 = np.random.choice(6, size=20000, replace=True, p=original_probabilities)

# Estimate the categorical distribution parameters from the data.
estimated_probabilities1 = mle_cat(r1, 6)
estimated_probabilities2 = mle_cat(r2, 6)

# Plot the original and the estimated models for comparison.

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.set_xlabel('lambda', fontsize=25)
ax1.set_ylabel('P(lambda)', fontsize=25)
ax1.set_ylim([0,0.35])
ax1.set_title('Original',fontsize=18)
ax1.bar(np.arange(len(original_probabilities)),original_probabilities,color='b')

ax2.set_xlabel('lambda', fontsize=25)
ax2.set_ylim([0,0.35])
ax2.set_title('Estimated from 200 data points',fontsize=18)
ax2.bar(np.arange(len(estimated_probabilities1)), estimated_probabilities1,color='r')

ax3.set_xlabel('lambda', fontsize=25)
ax3.set_ylim([0,0.35])
ax3.set_title('Estimated from 20000 data points',fontsize=18)
ax3.bar(np.arange(len(estimated_probabilities2)), estimated_probabilities2,color='r')

plt.show()
