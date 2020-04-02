import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import h5py

if __name__ == '__main__':

    #signal_name = 'multisine'
    signal_name = 'sinesweep'

    # In[Load dataset]
    u_name = 'uval_' + signal_name
    u_filename = u_name + '.mat'

    y_name = 'yval_' + signal_name
    y_filename = y_name + '.mat'

    u = scipy.io.loadmat(os.path.join("BoucWenFiles", "Test signals", "Validation signals", u_filename))[u_name].reshape(1, -1)
    y = scipy.io.loadmat(os.path.join("BoucWenFiles", "Test signals", "Validation signals", y_filename))[y_name].reshape(1, -1)

    fs = np.array([750.0])

    # In[Plot dataset]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(y[0, :])
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Displacement (mm)')
    ax[0].grid(True)
    ax[1].plot(u[0, :])
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Force (N)')
    ax[1].grid(True)

    # In[Save in an hdf file]

    # can only write a group once, delete file to re-write the same group
    filename = os.path.join('BoucWenFiles', 'Test signals', 'test.h5')
    hf = h5py.File(filename, 'a')
    ds_signal = hf.create_group(signal_name)  # signal group
    ds_signal.create_dataset('y', data=y.transpose())
    ds_signal.create_dataset('u', data=u.transpose())
    ds_signal.create_dataset('fs', data=fs)
    hf.close()
