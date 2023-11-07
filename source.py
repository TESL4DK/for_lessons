import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def func(t, r, W, W0, sigma, d, G, g):
    W[0] = W0[0]* np.e ** -((t - t0[0]) /2*sigma[0]) ** 2
    W[1] = W0[1]* np.e ** -((t - t0[1]) /2*sigma[1]) ** 2
    r = np.asarray(r, dtype=np.complex128)
    drdt = np.zeros_like(r, dtype=np.complex128)
    drdt[0] = G[3]*r[4] - 1j*W[0]*(r[3] - r[1])
    drdt[4] = -(G[3] + G[5])*r[4] - 1j*(W[0]*(r[1] - r[3]) + W[1]*(r[7] - r[5]))
    drdt[8] = G[5]*r[4] - 1j*W[1]*(r[5] - r[7])
    drdt[1] = (-g[1] + 1j*d[0])*r[1] - 1j*(W[0]*(r[4] - r[0]) - W[1]*r[2])
    drdt[2] = (-g[2] + 1j*(d[0] -d[1]))*r[2] - 1j*(W[0]*r[5] - W[1]*r[1])
    drdt[5] = (-g[5] - 1j*d[1])*r[5] - 1j*(W[1]*(r[8] - r[4]) + W[0]*r[2])
    drdt[3] = np.conj(drdt[1])
    drdt[6] = np.conj(drdt[2])
    drdt[7] = np.conj(drdt[5])
    return drdt

t0 = np.zeros(2)
t0[0] = 30
t0[1] = 20

sigma = np.zeros(2)
sigma[0] = 3
sigma[1] = 3

W0 = np.zeros(2)
W0[0] = 5
W0[1] = 5

G = np.zeros(7)
G[3] = 0.5
G[5] = 0.5

W = np.zeros(2)
W[0] = 5
W[1] = 5

g = np.zeros(6)
g[0] = 0.5
g[1] = 0.5

d = np.zeros(2)

r0 = [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]

t_span = [0.0, 50.0]
args = (W, W0, sigma, d, G, g)

sol = solve_ivp(func, t_span, r0, method="RK45", args=args)

r11 = sol.y[0, :]
r22 = sol.y[4, :]
r33 = sol.y[8, :]

plt.plot(sol.t, r11, label = "r11")
plt.plot(sol.t, r22, label = "r22")
plt.plot(sol.t, r33, label = "r33")

plt.legend()
plt.grid()
plt.show()