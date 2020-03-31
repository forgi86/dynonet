import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import numba as nb

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


def lfilter_mimo_channels_last(b, a, u_in):
    batch_size, seq_len, in_ch = u_in.shape
    out_ch, _, _ = a.shape
    y_out = np.zeros_like(u_in, shape=(batch_size, seq_len, out_ch))
    for out_idx in range(out_ch):
        for in_idx in range(in_ch):
            y_out[:, :, out_idx] += scipy.signal.lfilter(b[out_idx, in_idx, :], a[out_idx, in_idx, :],
                                                         u_in[:, :, in_idx], axis=-1)
    return y_out # [B, T, O]


def lfilter_mimo_components_channels_last(b, a, u_in):
    batch_size, seq_len, in_ch = u_in.shape
    out_ch, _, _ = a.shape
    y_comp_out = np.zeros_like(u_in, shape=(batch_size, seq_len, out_ch, in_ch))
    for out_idx in range(out_ch):
        for in_idx in range(in_ch):
            y_comp_out[:, :, out_idx, in_idx] = scipy.signal.lfilter(b[out_idx, in_idx, :], a[out_idx, in_idx, :], u_in[:, :, in_idx], axis=-1)
    return y_comp_out  # [B, T, O, I]



if __name__ == '__main__':

    in_ch = 3
    out_ch = 4
    n_b = 2
    n_a = 1

    batch_size = 16
    seq_len = 32

    # Coefficients of the polynomials
    b_coeff = np.random.randn(*(out_ch, in_ch, n_b))
    a_coeff = np.random.rand(*(out_ch, in_ch, n_a))

    # Polynomials
    a_poly = np.empty_like(a_coeff, shape=(out_ch, in_ch, n_a + 1))
    a_poly[:, :, 0] = 1
    a_poly[:, :, 1:] = a_coeff[:, :, :]
    b_poly = np.array(b_coeff)

    eps = 1e-6  # numerical perturbation

    # In[Filter with initial condition]
    y_0 = np.random.randn(*(out_ch, in_ch, n_a))
    u_0 = np.random.randn(*(out_ch, in_ch, n_b))
    u_in = 1*np.random.randn(*(batch_size, seq_len, in_ch))
    #y, _, _ = lfilter_ic(b_poly, a_poly, u_in, y_0, u_0)



    y_out = lfilter_mimo_channels_last(b_poly, a_poly, u_in)
    y_out_comp = lfilter_mimo_components_channels_last(b_poly, a_poly, u_in)
    y_out_2 = np.sum(y_out_comp, axis=-1)

    assert (np.allclose(y_out, y_out_2))

    # In[Sensitivity]
    d0_np = np.array([1.0])
    sens_b = np.zeros_like(u_in, shape=(batch_size, out_ch, in_ch, n_b, seq_len))
    for out_idx in range(out_ch):
        for in_idx in range(in_ch):
            sens_b[:, out_idx, in_idx, 0, :] = sp.signal.lfilter(a_poly[out_idx, in_idx, :], d0_np, u_in[:, in_idx])
    for idx_coeff in range(1, n_b):
        sens_b[:, :, :, idx_coeff, idx_coeff:] = sens_b[:, :, :, 0, :-idx_coeff]
    #sens_b = torch.as_tensor(sens_b) # B, O, I, D, T

    grad_out = np.random.randn(*(batch_size, out_ch, seq_len))
    grad_b = np.einsum('boidt,bot->oid', sens_b, grad_out)
    grad_bb = np.einsum('boidt,bqt->oid', sens_b, grad_out)

    #grad_bb = np.einsum('b...t,b...t', sens_b, grad_out)


