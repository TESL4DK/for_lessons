import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def f(t, ρ, Γ, Ω, γ, ω):
    du = np.zeros(ρ.shape, dtype=complex)
    np.zeros_like(ρ, dtype=np.complex128)
    du[0] = Γ[3] * ρ[4] - 1j * Ω[0] * (ρ[3] - ρ[1])
    du[4] = -(Γ[3] + Γ[5]) * ρ[4] - 1j * (Ω[0] * (ρ[1] - ρ[3]) + Ω[1] * (ρ[7] - ρ[5]))
    du[8] = Γ[5] * ρ[4] - 1j * Ω[1] * (ρ[5] - ρ[7])
    du[1] = (-γ[1] + 1j * ω[0]) * ρ[1] - 1j * (Ω[0] * (ρ[4] - ρ[0]) - Ω[1] * ρ[2])
    du[2] = (-γ[2] + 1j * (ω[0] - ω[1])) * ρ[2] - 1j * (Ω[0] * ρ[5] - Ω[1] * ρ[1])
    du[5] = (-γ[5] - 1j * ω[1]) * ρ[5] - 1j * (Ω[1] * (ρ[8] - ρ[4]) + Ω[0] * ρ[2])
    du[3] = np.conj(du[1])
    du[6] = np.conj(du[2])
    du[7] = np.conj(du[5])
    return du


Γ = np.zeros(7)
Γ[3] = 0.5
Γ[5] = 0.5

Ω = np.zeros(2)
Ω[0] = 5
Ω[1] = 5

γ = np.zeros(6)
γ[0] = 0.5
γ[1] = 0.5

ω = np.zeros(2)

ρ0 = [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]

t_span = [0.0, 50.0]

sol = solve_ivp(f, t_span, ρ0, method="RK45", args=(Γ, Ω, γ, ω))


sec1 = sol.y[0, :]
#sec2 = sol.y[1, :]
#sec3 = sol.y[2, :]
#sec4 = sol.y[3, :]
sec5 = sol.y[4, :]
#sec6 = sol.y[5, :]
#sec7 = sol.y[6, :]
#sec8 = sol.y[7, :]
sec9 = sol.y[8, :]

plt.plot(sol.t, sec1)
#plt.plot(sol.t, sec2)
#plt.plot(sol.t, sec3)
#plt.plot(sol.t, sec4)
plt.plot(sol.t, sec5)
#plt.plot(sol.t, sec6)
#plt.plot(sol.t, sec7)
#plt.plot(sol.t, sec8)
plt.plot(sol.t, sec9)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Second Component of ρ Over Time')
plt.grid()
plt.show()
