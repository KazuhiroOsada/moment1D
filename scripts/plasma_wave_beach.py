import numpy as np
import matplotlib.pyplot as plt

from reader import reader


if __name__ == "__main__":
    r = reader(prefix="../results/plasma_wave_beach/")
    r.read_parameters()

    m = 1.0
    q = -1.0
    dt = r.dt
    diag_interval = r.diag_interval
    max_iters = r.max_iters

    iters = list(range(0, max_iters, diag_interval))
    t = np.array(iters) * dt

    it = iters[-1]

    U_field = r.read_field(it=it)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(f"Iteration {it} (t={t[it//diag_interval]:.2f})")

    axes[0].plot(U_field[0], label="Ex")
    axes[0].plot(U_field[1], label="Ey")
    axes[0].plot(U_field[2], label="Ez")
    axes[0].legend()
    
    U_moment = r.read_moment("electrons", it=it) 
    axes[1].plot(U_moment[0], label="Rho")
    axes[1].plot(U_moment[1], label="Ux")
    axes[1].plot(U_moment[2], label="Uy")
    axes[1].plot(U_moment[3], label="Uz")
    axes[1].plot(U_moment[4], label="ENE")
    axes[1].legend()

    plt.show()
    plt.close()
