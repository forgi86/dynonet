# dynoNet: a neural network architecture for learning of dynamical systems 


The following fitting methods for neural dynamical models are implemented and tested

 1. Full simulation error minimization
 2. Truncated simulation error minimization
 3. Soft-constrained integration
 4. One-step prediction error minimization


# Folders:
* [torchid](torchid):  PyTorch implementation of the linear dynamical operator (aka G-block in the paper) used in dynoNet
* [examples](examples): examples using dynoNet for system identification 
* [common](common): definition of metrics R-square, RMSE, fit index 

Three [examples](examples) discussed in the paper are:

* [WH2009](examples/WH2009): A circuit with Wiener-Hammerstein behavior. Experimental dataset from http://www.nonlinearbenchmark.org
* [BW](examples/WB): Bouc-Wen. A nonlinear dynamical system describing hysteretic effects in mechanical engineering. Experimental dataset from http://www.nonlinearbenchmark.org
* [EMPS](examples/EMPS): A controlled prismatic joint (Electro Mechanical Positioning System). Experimental dataset from http://www.nonlinearbenchmark.org

For the [WH2009](examples/WH2009) example, the main scripts are:

 *  ``WH2009_train.py``: Training of the dynoNet model on the training dataset
 *  ``WH2009_test.py``: Testing of the dynoNet model on the test dataset, computation of metrics.
  
Similar scripts are provided for the other examples.

# Software requirements:
Simulations were performed on a Python 3.7 conda environment with

 * numpy
 * scipy
 * matplotlib
 * pandas
 * pytorch (version 1.4)
 
These dependencies may be installed through the commands:

```
conda install numpy scipy pandas matplotlib
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
