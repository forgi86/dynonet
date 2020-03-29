import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt


def lfilter_ic(b_poly, a_poly, u_in, y_0=None, u_0=None):

    if y_0 is None and u_0 is None:
        z_init = None
    else:
        z_init = scipy.signal.lfiltic(b_poly, a_poly, y_0, u_0)
    if z_init is not None:
        y, z_end = scipy.signal.lfilter(b_poly, a_poly, u_in, zi=z_init)
    else:
        y = scipy.signal.lfilter(b_poly, a_poly, u_in, zi=z_init)
    z_init = None
    z_end = None
    return y, z_init, z_end

if __name__ == '__main__':

    # Coefficients of the polynomials
    b_coeff = np.array([0.02, 0.03, 0.04])  # b_0, b_1, b_2
    a_coeff = np.array([-1.87212998940304, 0.942776404097492])  # a_1, a_2

    # Polynomials
    a_poly = np.r_[1.0, a_coeff]
    b_poly = np.array(b_coeff)

    eps = 1e-6  # numerical perturbation

    # In[Filter with initial condition]
    y_0 = np.array([1.0, 2.0]) # y_-1, y_-2
    u_0 = np.array([3.0, 4.0]) # u_-1, u_-2
    u_in = 0*np.random.randn(150)
    y, _, _ = lfilter_ic(b_poly, a_poly, u_in, y_0, u_0)


    # Equivalent initial condition, with no input
    y_0_bar = np.zeros_like(y_0)
    y_m1_bar = y_0[0] - b_poly[2]/a_poly[2]*u_0[0]
    y_m2_bar = y_0[1] + (a_poly[1]*b_poly[2]/(a_poly[2] ** 2) - b_poly[1] / a_poly[2]) * u_0[0] - b_poly[2] / a_poly[2] * u_0[1]

    y_0_bar[0] = y_0[0] - b_poly[2]/a_poly[2]*u_0[0]
    #y_0_bar[1] = y_0[1] -b_poly[1]/a_poly[2]*u_0[0] - b_poly[2]/a_poly[2]*u_0[1] + a_poly[1]*b_poly[2]/(a_poly[2]**2)*u_0[0]
    y_0_bar[1] = y_0[1] + (a_poly[1]*b_poly[2]/(a_poly[2] ** 2) - b_poly[1] / a_poly[2]) * u_0[0] - b_poly[2] / a_poly[2] * u_0[1]

    # Verify equivalent initial condition
    zi = scipy.signal.lfiltic(b_poly, a_poly, y_0, u_0)
    zi_bar = scipy.signal.lfiltic(b_poly, a_poly, y_0_bar, 0*u_0)

    # In[Free response]
    delta = a_poly[1]**2 -4*a_poly[2]
    if delta < 0:
        r = -a_poly[1]

    # In[Analytical sensitivities b]

    sens_b0_an, _, _ = lfilter_ic([1.0], a_poly, u_in, [0.0], u_0)  # this is correct!
    sens_b1_an, _, _ = lfilter_ic([0.0, 1.0], a_poly, u_in, [0.0], u_0)  # this is correct!
    sens_b2_an, _, _ = lfilter_ic([0.0, 0.0, 1.0], a_poly, u_in, [0.0], u_0)  # this is correct!

    # In[Analytical sensitivities a]

    sens_a1_an, _, _ = lfilter_ic([0.0, 1.0], a_poly, -y, [0.0], -y_0)  # this is correct!
    sens_a2_an, _, _ = lfilter_ic([0.0, 0.0, 1.0], a_poly, -y, [0.0], -y_0)  # this is correct!

    # In[Perturbation on coefficients b]
    # b0
    b_coeff_eps = np.array(b_coeff)
    b_coeff_eps[0] += eps
    b_poly_eps = np.array(b_coeff_eps)
    y_eps, _ ,_ = lfilter_ic(b_poly_eps, a_poly, u_in, y_0, u_0)
    sens_b0_num = (y_eps - y) / eps

    # b1
    b_coeff_eps =  np.array(b_coeff)
    b_coeff_eps[1] += eps
    b_poly_eps = np.array(b_coeff_eps)
    y_eps, _, _ = lfilter_ic(b_poly_eps, a_poly, u_in, y_0, u_0)
    sens_b1_num = (y_eps - y) / eps

    # b2
    b_coeff_eps =  np.array(b_coeff)
    b_coeff_eps[2] += eps
    b_poly_eps = np.array(b_coeff_eps)

    y_eps, _, _ = lfilter_ic(b_poly_eps, a_poly, u_in, y_0, u_0)
    sens_b2_num = (y_eps - y) / eps

    # In[Perturbation on coefficients a]
    # a1
    a_coeff_eps =  np.array(a_coeff)
    a_coeff_eps[0] += eps
    a_poly_eps = np.r_[1.0, a_coeff_eps]
    y_eps, _, _ = lfilter_ic(b_poly, a_poly_eps, u_in, y_0, u_0)
    sens_a1_num = (y_eps - y) / eps

    # a2
    a_coeff_eps =  np.array(a_coeff)
    a_coeff_eps[1] += eps
    a_poly_eps = np.r_[1.0, a_coeff_eps]
    y_eps, _, _ = lfilter_ic(b_poly, a_poly_eps, u_in, y_0, u_0)
    sens_a2_num = (y_eps - y) / eps


    # In[Output]
    plt.figure()
    plt.plot(y, '*')

    # In[b sensitivity]
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(sens_b0_num, 'b', label='$b_0$ num')
    ax[0].plot(sens_b0_an, 'r', label='$b_0$ an')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(sens_b1_num, 'b', label='$b_1$ num')
    ax[1].plot(sens_b1_an, 'r', label='$b_1$ an')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(sens_b2_num, 'b', label='$b_2$ num')
    ax[2].plot(sens_b2_an, 'r', label='$b_2$ an')
    ax[2].legend()
    ax[2].grid()

    # In[2]
    plt.figure()
    plt.plot(sens_b0_num[0:-2], label='$b_0$')
    plt.plot(sens_b1_num[1:-1], label='$b_1 q^1$')
    plt.plot(sens_b2_num[2:], label='$b_2 q^2$')
    plt.grid()
    plt.legend()

    # In[2]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(sens_a1_num, 'b', label='$a_1$ num')
    ax[0].plot(sens_a1_an, 'r', label='$a_1$ an')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(sens_a2_num, 'b', label='$a_2$ num')
    ax[1].plot(sens_a2_an, 'r', label='$a_2$ an')
    ax[1].legend()
    ax[1].grid()


    # In[2]
    plt.figure()
    plt.plot(sens_a1_num[0:-1], label='$a_1$')
    plt.plot(sens_a2_num[1:], label='$a_2q^1$')
    plt.legend()
    plt.grid()
