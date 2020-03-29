import numpy as np
import matplotlib.pyplot as plt


def sigmoid(val):
    return 1/(1 + np.exp(-val))

def stable_coeff(rho, psi):
    r = sigmoid(rho)
    theta = np.pi * sigmoid(psi)

    a_1 = -2*r*np.cos(theta)
    a_2 = r**2
    return a_1, a_2

def roots_polynomial(a_1, a_2):
    delta = a_1**2 - 4 * a_2
    delta = delta.astype(np.complex)
    root_1 = (-a_1 + np.sqrt(delta))/2
    root_2 = (-a_1 - np.sqrt(delta))/2
    idx_real = delta > 0
    return root_1, root_2, idx_real


if __name__ == '__main__':

    N = 100000
    rho = np.random.randn(N)*1
    psi = np.random.randn(N)*1

    a_1, a_2 = stable_coeff(rho, psi)
    r_1, r_2, idx_real = roots_polynomial(a_1, a_2)


    fig, ax = plt.subplots()
    ax.plot(a_1, a_2, '*')
    ax.plot(a_1[idx_real], a_2[idx_real], 'k*')
    ax.set_xlabel('a_1')
    ax.set_ylabel('a_2')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])


    fig, ax = plt.subplots()
    ax.plot(np.real(r_1), np.imag(r_1), 'r*')
    ax.plot(np.real(r_2), np.imag(r_2), 'r*')
    ax.plot(np.real(r_1)[idx_real], np.imag(r_1)[idx_real], 'k*')
    ax.plot(np.real(r_2)[idx_real], np.imag(r_2)[idx_real], 'k*')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])

    perc_real = np.sum(idx_real) / N *100
    print(f"Real poles in {perc_real:.1f} cases")
