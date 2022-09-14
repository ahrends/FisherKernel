# Fisher Kernel

Preliminary code for Fisher Kernel

Code to construct Fisher Kernels from HMMs obtained from the HMM-MAR toolbox (https://github.com/OHBA-analysis/HMM-MAR) to predict phenotypes from models of brain dynamics. 

See FisherKernel_example.m for examples. Script includes examples for Gaussian HMM and TDE-HMM, using the Fisher kernel in a SVM and in kernel ridge regression. 

List of functions:
./main contains main functions for Fisher Kernel: hmm_kernel.m constructs a kernel (e.g. Fisher kernel) and the corresponding feature matrix from an HMM, hmm_gradient.m only computes the feature matrix (e.g. Fisher score/gradient).

./utils contains helper functions, namely hmmdual_FK.m for dual estimation to work for Fisher kernel construction (within hmm_gradient.m and hmm_kernel.m) and transform_dataHMM.m which can be used to check the embedding used in the HMM and the features or kernel

./prediction contains preliminary functions for prediction models that use the Fisher kernel, for now krr_predict_FK.m, which runs kernel ridge regression using the Fisher kernel. cvfolds_FK.m is used for cross validation in krr_predict_FK.m.

Reference: Ahrends & Vidaurre (in prep.) Fisher kernel method to predict phenotypes from models of brain dynamics. 
