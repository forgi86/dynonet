An example inspired from the paper [1]]

It is about training of a Wiener system (G-F) with noise entering in the system before the non-linearity.
In such a setting, the non-linear least square estimate is generally biased.

In order to deal with this example, we perform an approximate Maximum Likelihood (ML) estimate.
We approximate the integral (27) in [1] using the rectangular integration rule and differentiate through
using plain back-propagation (see W_train_ML_refine.py)

We use the non-linear least square estimate (See W_train_NLS.py)
to initialize the estimate for the heavier ML estimation task.

To run the example:

1. install pyro, e.g., "pip install pyro-ppl"
2. python W_train_NLS_no_noise.py
3. python W_train_NLS.py
4. python W_train_ML_refine.py
5. python W_test.py

[1] A. Hagenblad, L. Ljung, and A. Wills. Maximum likelihood identification of Wiener models.  Automatica, 44 (2008) 2697–2705
