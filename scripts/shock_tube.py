import numpy as np
import matplotlib.pyplot as plt

from reader import reader


if __name__ == "__main__":
    r = reader(prefix="../results/shock_tube/")
    r.read_parameters()

    it = 20000
    U0 = r.read_moment("ions", it=0)
    U = r.read_moment("ions", it=it)
    x = np.arange(r.Nx)
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    axes[0].plot(x, U0[0], label="rho (t=0)", linestyle="dashed")
    axes[0].plot(x, U[0], label=f"rho (t={it})")
    axes[0].legend()
    axes[0].set_title("Density")

    axes[1].plot(x, U0[1], label="ux (t=0)", linestyle="dashed")
    axes[1].plot(x, U[1], label=f"ux (t={it})")
    axes[1].plot(x, U0[2], label="uy (t=0)", linestyle="dashed")
    axes[1].plot(x, U[2], label=f"uy (t={it})")
    axes[1].plot(x, U0[3], label="uz (t=0)", linestyle="dashed")
    axes[1].plot(x, U[3], label=f"uz (t={it})")
    axes[1].legend()
    axes[1].set_title("Velocity")

    axes[2].plot(x, U0[4], label="pressure (t=0)", linestyle="dashed")
    axes[2].plot(x, U[4], label=f"pressure (t={it})")
    axes[2].legend()
    axes[2].set_title("Pressure")

    plt.show()
       