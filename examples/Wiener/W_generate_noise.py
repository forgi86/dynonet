import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py


#a = [1.0, 0.3, -0.3]
#b = [0.0, -0.3, 0.3]
#c = []

a = [1, 0.5]
b = [0.0, 1.0]
c = [1.0, 1.0, 0.0]

var_w = 4.0
var_e = 1.0


def static_nl(y_lin):
    y_nl = np.polyval(c, y_lin) #c0 + c1*y_lin + c2*y_lin**2
    return y_nl


if __name__ == '__main__':

    n_real = 50
    N = 1000
    add_noise = True # add process noise
    output_filename = 'dataset.h5'
    dataset_name = 'train_noise'

    # In[]
    var_w = add_noise*var_w
    var_e = var_e
    std_w = np.sqrt(var_w)
    std_e = np.sqrt(var_e)

    # In[Wiener with noise model]
    u = np.random.randn(n_real, N)
    x0 = scipy.signal.lfilter(b, a, u, axis=-1)
    w = std_w*np.random.randn(n_real, N)
    x = x0 + w
    y0 = static_nl(x)
    e = std_e*np.random.randn(n_real, N)
    y = y0+e


    # In[Plot]
    plt.figure()
    plt.plot(y[0, :], 'r', label='y')
    plt.plot(y0[0, :], 'g', label='y0')
    plt.legend()

    plt.figure()
    plt.plot(x[0, :], 'g', label='x')
    plt.plot(x0[0, :], 'r', label='x0')
    plt.legend()

    # In[Save]
    if not (os.path.exists('data')):
        os.makedirs('data')
    filename = os.path.join('data', output_filename)
    hf = h5py.File(filename, 'a')
    ds_signal = hf.create_group(dataset_name)  # signal group
    ds_signal.create_dataset('u', data=u[..., None])
    ds_signal.create_dataset('x0', data=x0[..., None])
    ds_signal.create_dataset('w', data=w[..., None])
    ds_signal.create_dataset('y0', data=y0[..., None])
    ds_signal.create_dataset('y', data=y[..., None])
    hf.close()


