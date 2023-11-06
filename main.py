import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def func(t, r, W, d, G, g):
    #arg = -(t - t0)**2/sigma
    #W = W0 * exp**arg

    r = np.asarray(r, dtype=np.complex128)
    drdt = np.zeros_like(r, dtype=np.complex128)
    drdt[0] = G[3]*r[4] - 1j*W[0]*(r[3] - r[1])
    drdt[4] = -(G[3]+ G[5])*r[4] - 1j*(W[0]*(r[1] - r[3]) + W[1]*(r[7] - r[5]))
    drdt[8] = G[5]*r[4] - 1j*W[1]*(r[5] - r[7])
    drdt[1] = (-g[1] + 1j*d[0])*r[1] - 1j*(W[0]*(r[4] - r[0]) - W[1]*r[2])
    drdt[2] = (-g[2] + 1j*(d[0] -d[1]))*r[2] - 1j*(W[0]*r[5] - W[1]*r[1])
    drdt[5] = (-g[5] - 1j*d[1])*r[5] - 1j*(W[1]*(r[8] - r[4]) + W[0]*r[2])
    drdt[3] = np.conj(drdt[1])
    drdt[6] = np.conj(drdt[2])
    drdt[7] = np.conj(drdt[5])
    return drdt



G = np.zeros(7)
r = np.zeros(9)
W = np.zeros(2)
d = np.zeros(2)

r0 = [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]

W[0] = 5
W[1] = 5

G[3] = 0.5
G[5] = 0.5
g = G / 2

t_span = [0.0, 50.0]
args = (W, d, G, g)

sol = solve_ivp(func, t_span, r0, method="RK45", args=args)

r1 = sol.y[0, :]
#r2 = sol.y[1, :]
#r3 = sol.y[2, :]
#r4 = sol.y[3, :]
r5 = sol.y[4, :]
#r6 = sol.y[5, :]
#r7 = sol.y[6, :]
#r8 = sol.y[7, :]
r9 = sol.y[8, :]

plt.plot(sol.t, r1, label = "r1")
#plt.plot(sol.t, r2, label = "r2")
#plt.plot(sol.t, r3, label = "r3")
#plt.plot(sol.t, r4, label = "r4")
plt.plot(sol.t, r5, label = "r5")
#plt.plot(sol.t, r6, label = "r6")
#plt.plot(sol.t, r7, label = "r7")
#plt.plot(sol.t, r8, label = "r8")
plt.plot(sol.t, r9, label = "r9")
plt.legend()
plt.grid()
plt.show()