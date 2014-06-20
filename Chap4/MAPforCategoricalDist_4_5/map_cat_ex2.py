import matplotlib.pyplot as plt
import numpy as np
import os, sys

from map_cat import map_cat

# Generate random values from the categorical distribution 
# with 6 categories and the corresponding probabilities.
original_probabilities = [0.25, 0.15, 0.1, 0.1, 0.15, 0.25]
r1 = np.random.choice(6, size=200,   replace=True, p=original_probabilities)
r2 = np.random.choice(6, size=2000,  replace=True, p=original_probabilities)
r3 = np.random.choice(6, size=20000, replace=True, p=original_probabilities)

# MAP estimate of the categorical distribution parameters from the data.
prior = [10, 100, 1000, 1000, 100, 10]
estimated_probabilities_map1 = map_cat(r1, prior)
estimated_probabilities_map2 = map_cat(r2, prior)
estimated_probabilities_map3 = map_cat(r3, prior)

# Plot the original and the estimated models for comparison.
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
ax1.set_xlabel('lambda', fontsize=25)
ax1.set_ylabel('P(lambda)', fontsize=25)
ax1.set_ylim([0,0.45])
ax1.set_title('Original',fontsize=18)
ax1.bar(np.arange(len(original_probabilities)),original_probabilities,color='b')

ax2.set_xlabel('lambda', fontsize=25)
ax2.set_ylim([0,0.45])
ax2.set_title('MAP estimate with 200 data points', fontsize=12)
ax2.bar(np.arange(len(estimated_probabilities_map1)), estimated_probabilities_map1, color='r')

ax3.set_xlabel('lambda', fontsize=25)
ax3.set_ylim([0,0.45])
ax3.set_title('MAP Estimate with 2000 data points',fontsize=12)
ax3.bar(np.arange(len(estimated_probabilities_map2)), estimated_probabilities_map2, color='r')

ax4.set_xlabel('lambda', fontsize=25)
ax4.set_ylim([0,0.45])
ax4.set_title('MAP Estimate with 200000 data points',fontsize=12)
ax4.bar(np.arange(len(estimated_probabilities_map3)), estimated_probabilities_map3, color='r')

plt.show()
