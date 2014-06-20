import matplotlib.pyplot as plt
import numpy as np
import os, sys

dirname = os.path.dirname
cwd = dirname(os.path.realpath(__file__))
sys.path.append(dirname(dirname(cwd)))

from Chap4.MAPforCategoricalDist_4_5.map_cat import map_cat
from by_cat import by_cat

# Generate random values from the categorical distribution 
# with 6 categories and the corresponding probabilities.
original_probabilities = [0.25, 0.15, 0.1, 0.1, 0.15, 0.25]
r = np.random.choice(6, size=100000, replace=True, p=original_probabilities)

# MAP estimate of the categorical distribution parameters from the data.
prior = [1, 1, 1, 1, 1, 1]
(alpha_post, prediction) = by_cat(r, prior)

# MLE estimate of the categorical distribution parameters from the data.
estimated_probabilities_map = map_cat(r, prior)

# Plot the original and the estimated models for comparison.
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.set_xlabel('lambda', fontsize=25)
ax1.set_ylabel('P(lambda)', fontsize=25)
ax1.set_ylim([0,0.35])
ax1.set_title('Original',fontsize=18)
ax1.bar(np.arange(len(original_probabilities)), original_probabilities,color='g')

ax2.set_xlabel('lambda', fontsize=25)
ax2.set_ylim([0,0.35])
ax2.set_title('MAP estimate with 100000 points', fontsize=16)
ax2.bar(np.arange(len(estimated_probabilities_map)), estimated_probabilities_map,color='b')

ax3.set_xlabel('lambda', fontsize=25)
ax3.set_ylim([0,0.35])
ax3.set_title('Bayesian estimate with 100000 points',fontsize=16)
ax3.bar(np.arange(len(prediction)), prediction,color='r')

plt.show()
