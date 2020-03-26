import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

allData = np.random.rand(20,80,200)

#allData is a 64x256x913 array


N = 100
n_batch = 2
n_b = 3
grad_out = np.random.rand(N + n_b - 1, n_batch)
sens_b = np.random.rand(N, 1)
z = scipy.signal.correlate(grad_out, sens_b, 'valid')
