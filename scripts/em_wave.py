import numpy as np
import matplotlib.pyplot as plt

from reader import reader


if __name__ == "__main__":
    r = reader(prefix="../results/em_wave/")
    r.read_parameters()

    it = 3000
    U0 = r.read_field(it=0)
    U = r.read_field(it=it)
    x = np.arange(r.Nx)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(x, U0[0], label="Ex (t=0)", linestyle="dashed")
    axes[0].plot(x, U[0], label=f"Ex (t={it})")
    axes[0].plot(x, U0[1], label="Ey (t=0)", linestyle="dashed")
    axes[0].plot(x, U[1], label=f"Ey (t={it})")
    axes[0].plot(x, U0[2], label="Ez (t=0)", linestyle="dashed")
    axes[0].plot(x, U[2], label=f"Ez (t={it})")
    axes[0].legend()
    axes[0].set_title("Electric Field")
    
    axes[1].plot(x, U0[3], label="Bx (t=0)", linestyle="dashed")
    axes[1].plot(x, U[3], label=f"Bx (t={it})")
    axes[1].plot(x, U0[4], label="By (t=0)", linestyle="dashed")
    axes[1].plot(x, U[4], label=f"By (t={it})")
    axes[1].plot(x, U0[5], label="Bz (t=0)", linestyle="dashed")
    axes[1].plot(x, U[5], label=f"Bz (t={it})")
    axes[1].legend()
    axes[1].set_title("Magnetic Field")

    plt.show()
