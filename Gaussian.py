import numpy as np
import matplotlib.pyplot as plt

W0 = 10
t = np.arange(1, 41)
t0 = 20
sigma = 3

W = W0*np.exp(-(((t - t0)/(2*sigma))**2))

plt.plot(W)
plt.grid()
plt.show()
