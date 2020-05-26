import numpy as np
import matplotlib
import matplotlib.pyplot as plt


if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 12})
    matplotlib.rc('text', usetex=True)

    # Plotting the different regions for a system with denominator z**2 + a1*z + a2

    a1 = np.linspace(-2, 2, 1000)  # range for parameter a1
    a2_complex = a1**2/4  # complex conjugate poles for a2> a2_complex
    a2_stab_min = np.abs(a1) - 1
    a2_stab_max = np.ones_like(a1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(a1, a2_stab_min, 'k')
    ax.plot(a1, a2_stab_max, 'k')
    ax.fill_between(a1, a2_complex, a2_stab_max, facecolor='b', alpha=0.2,  label='stable complex conjugate poles')
    ax.fill_between(a1, a2_stab_min, a2_complex, label='stable real poles')
    ax.plot(a1, a2_complex, 'k--', label='stable real coincident poles')
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.0, 1.5])
    ax.set_xlabel('$a_1$', fontsize=16)
    ax.set_ylabel('$a_2$', fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig('stable_2ndorder.pdf')




