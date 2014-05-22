import numpy as np
import matplotlib.pyplot as plt

#params = (1000, 1000, 1000, 1000, 1000, 1000)
params = (10, 100, 1000, 1000, 100, 10)
#params = (1, 1, 1, 1, 1, 1)
s = np.random.dirichlet(params, 3).transpose()

fig = plt.figure()

colour = 'b'
ax1 = fig.add_subplot(131)
ind = np.arange(len(s[:,[0]]))  # the x locations for the groups
width = 0.55
ax1.bar(ind, s[:,[0]].T[0], width, color=colour)
ax1.set_xlabel('lambda', fontsize=20, color='red')
ax1.set_ylabel('p(lambda)', fontsize=20, color='red')
ax1.set_title('Params ' + repr(params))

ax2 = fig.add_subplot(132)
ind = np.arange(len(s[:,[1]]))  # the x locations for the groups
width = 0.55
ax2.bar(ind, s[:,[1]].T[0], width, color=colour)
ax2.set_xlabel('lambda', fontsize=20, color='r')
ax2.set_title('Params ' + repr(params))

ax3 = fig.add_subplot(133)
ind = np.arange(len(s[:,[2]]))  # the x locations for the groups
width = 0.55
ax3.bar(ind, s[:,[2]].T[0], width, color=colour)
ax3.set_xlabel('lambda', fontsize=20, color='r')
ax3.set_title('Params ' + repr(params))
plt.show()
