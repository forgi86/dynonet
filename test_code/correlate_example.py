import numpy as np
import scipy

from scipy import signal

SIG = np.zeros((3, 8*128))

sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
sig_noise = sig + 0.1*np.random.randn(len(sig))
corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128

SIG[0, :] = np.copy(sig)
SIG[1, :] = np.copy(sig_noise)
SIG[2, :] = np.copy(sig_noise)

CORR = signal.correlate(SIG, np.ones((1, 128)), mode='same') / 128

import matplotlib.pyplot as plt
clock = np.arange(64, len(sig), 128)
fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(sig)
ax_orig.plot(clock, sig[clock], 'ro')
ax_orig.set_title('Original signal')
ax_noise.plot(sig_noise)
ax_noise.set_title('Signal with noise')
ax_corr.plot(corr)
ax_corr.plot(clock, corr[clock], 'ro')
ax_corr.axhline(0.5, ls=':')
ax_corr.set_title('Cross-correlated with rectangular pulse')
ax_orig.margins(0, 0.1)
fig.tight_layout()
fig.show()


fig,ax = plt.subplots()
plt.plot(CORR.T)