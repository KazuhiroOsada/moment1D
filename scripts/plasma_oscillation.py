import numpy as np
import matplotlib.pyplot as plt

from reader import reader


def compute_total_energy(rho, ux, uy, uz, Ex, Ey, Ez, m, q, epsilon0=1.0):
    n = rho / m
    omega_p_sq = (q * q * n) / (epsilon0 * m)
    Jx = (q / m) * rho * ux
    Jy = (q / m) * rho * uy
    Jz = (q / m) * rho * uz
    J_sq = Jx * Jx + Jy * Jy + Jz * Jz
    E_sq = Ex * Ex + Ey * Ey + Ez * Ez
    energy_density = 0.5 * J_sq / (epsilon0 * omega_p_sq) + 0.5 * epsilon0 * E_sq
    return np.mean(energy_density)


if __name__ == "__main__":
    r = reader(prefix="../results/plasma_oscillation/")
    r.read_parameters()

    m = 1.0
    q = -1.0
    dt = r.dt
    diag_interval = r.diag_interval
    max_iters = r.max_iters

    iters = list(range(0, max_iters, diag_interval))
    t = np.array(iters) * dt

    Ex_series = []
    Ey_series = []
    Ez_series = []
    energy_series = []

    for it in iters:
        U_field = r.read_field(it=it)
        U_moment = r.read_moment("electrons", it=it)

        Ex = U_field[0]
        Ey = U_field[1]
        Ez = U_field[2]

        Ex_series.append(np.mean(Ex))
        Ey_series.append(np.mean(Ey))
        Ez_series.append(np.mean(Ez))

        rho = U_moment[0]
        ux = U_moment[1]
        uy = U_moment[2]
        uz = U_moment[3]
        energy_series.append(compute_total_energy(rho, ux, uy, uz, Ex, Ey, Ez, m, q))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, Ex_series, label="Ex")
    axes[0].plot(t, Ey_series, label="Ey")
    axes[0].plot(t, Ez_series, label="Ez")
    axes[0].legend()
    axes[0].set_title("Electric Field (spatial mean)")

    axes[1].plot(t, energy_series, label="Total energy")
    axes[1].set_ylim(0.0, energy_series[0] * 1.1)
    axes[1].legend()
    axes[1].set_title("Energy conservation")
    axes[1].set_xlabel("Time")

    plt.tight_layout()
    plt.show()
