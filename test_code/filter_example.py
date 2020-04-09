import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import numba as nb
import control

if __name__ == '__main__':

    in_channels = 3
    out_channels = 4
    n_a = 0
    n_b = 10
    ts = 1.0

    a_coeff = np.random.randn(out_channels, in_channels, n_a)
    b_coeff = np.random.randn(out_channels, in_channels, n_b)

    a_poly = np.zeros_like(a_coeff, shape=(out_channels, in_channels, n_a+1))
    a_poly[:, :, 0] = 1.0
    b_poly = np.array(b_coeff)

    M = n_b  # numerator coefficients
    N = n_a + 1  # denominator coefficients
    if M > N:
        num = b_poly
        den = np.c_[a_poly, np.zeros((out_channels, in_channels, M-N))]
    elif N > M:
        nun = np.c_[b_poly, np.zeros((out_channels, in_channels, N-M))]
        den = a_poly
    else:
        num = b_poly
        den = a_poly

    G = scipy.signal.TransferFunction(num[0, 0, :], den[0, 0, :], dt=ts)
    Gg = control.TransferFunction(num[0, 0, :], den[0, 0, :], ts)

    G_MIMO = control.TransferFunction(num, den, ts)

    len_imp = n_b
    t, y_imp = control.impulse_response(Gg, np.arange(len_imp)*ts)

