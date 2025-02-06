# Fisher Kernel

Code for Fisher kernel and other methods to predict from Hidden Markov Models.

The Fisher kernel (Jaakkola & Haussler, 1998) is a framework for combining a discriminative with a generative model, making it a mathematically principled approach to using a Hidden Markov Model for machine learning. Beside the main functions, this repository contains code to reproduce [Ahrends, Woolrich, & Vidaurre (2024) elife](https://elifesciences.org/articles/95125), where we use the Fisher kernel and other methods to predict individual traits from models of brain dynamics. There are also examples in Matlab & Python.

Dependencies:
HMM-MAR toolbox for HMM in Matlab - https://github.com/OHBA-analysis/HMM-MAR/
GLHMM toolbox for HMM in Python - https://github.com/vidaurre/glhmm
(Other toolboxes used for static FC methods in the paper: NetsPredict https://github.com/vidaurre/NetsPredict and covariance toolbox https://github.com/alexandrebarachant/covariancetoolbox)

NOTE: Main Fisher kernel and prediction functionality is included in the HMM-MAR toolbox in Matlab and in the GLHMM toolbox in Python.


## Quick start

In Python, install the GLHMM toolbox in your environment:
```
pip install glhmm
```

Then fit an HMM and construct the Fisher kernel from it:
```python
from glhmm import glhmm, prediction

# initialise and train an HMM
hmm = glhmm.glhmm(model_beta='no', K=4, covtype='full') # for a Gaussian HMM with 4 states
hmm.train(X=None, Y=timeseries, indices=T) # where timeseries are a numpy array containing the n_subjects*n_timepoints, n_parcels data to be modelled, and T contains the session start and stop indices

# predict a variable (e.g. individual subject traits) from the HMM
options = {}
options['nfolds'] = 10 # number of folds for inner & outer CV loops
options['shape'] = 'linear'
options['incl_Mu'] = True # include state means
options['incl_Sigma'] = True # include state covariances
# the initial state probabilities and the transition probabilities
# are used by default, but can be excluded by setting options['incl_Pi']
# and options['incl_P'], respectively, to False
# use HMM trained/loaded above and standardised timeseries to predict subjects' age:
results = prediction.predict_phenotype(hmm, timeseries, behav, T, predictor='Fisherkernel', estimator='KernelRidge', options=options)
# results['behav_pred'] contains the predicted variable
```

## Repository overview

*./elife2024 contains all functions and scripts to replicate the paper using the HCP resting-state fMRI data.

*./examples contains examples in Matlab and Python (coming soon). Examples include Fisher kernel for Gaussian and TDE-HMM in different regression/classification problems.

*./main contains main functions for Fisher Kernel: hmm_kernel.m constructs a kernel (e.g. Fisher kernel) and the corresponding feature matrix from an HMM, hmm_gradient.m only computes the feature matrix (e.g. Fisher score/gradient).

   *./main/utils contains helper functions, namely hmmdual_FK.m for dual estimation to work for Fisher kernel construction (within hmm_gradient.m and hmm_kernel.m) and transform_dataHMM.m which can be used to check the embedding used in the HMM and the features or kernel

   *./main/prediction contains functions for kernel regression, for now predictPhenotype_kernels.m and krr_predict_FK.m, both run kernel ridge regression using output from hmm_kernel. cvfolds_FK.m is used for randomised cross validation, optionally accounting for family structure. predictPhenotype_kernels.m and predictPhenotypes_kernels_kfolds.m (the fold-level equivalent of predictPhenotype_kernels.m) also use deconfounding.

## Reference

If you use this repository, please cite:
Ahrends, Woolrich, & Vidaurre (2024) Predicting individual traits from models of brain dynamics accurately and reliably using the Fisher kernel. elife. doi: https://doi.org/10.7554/eLife.95125 
