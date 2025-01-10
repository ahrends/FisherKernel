function [Kernel, features, D] = build_kernels_PCAstates(datadir, hmmdir, kerneldir, HMM_name, Fn, Kn)
% [Kernel, features, D] = build_kernels_PCAstates(datadir, hmmdir, kerneldir, HMM_name, Fn, Kn)
% 
% build kernels from HMMs using PCA-reduced state parameters and transition
% probabilities/initial state probabilities
% (wrapper for hmm_kernel)
%
% Input: 
%    datadir: directory for HCP rsFMRI timecourses
%    hmmdir: directory where pre-trained HMM can be found
%    kerneldir: (output) directory where kernels and features will be saved
%    HMM_name: file name of the HMM to be loaded
%    Fn: select embedding: 1 for gradient embedding (Fisher kernel), 
%        2 for naive (no embedding), 3 for naive norm (normalised 
%        parameters across subjects)
%    Kn: select kernel shape: 1 for linear, 2 for Gaussian
% 
% Output (will be written to kerneldir):
%    Kernel: the samples x samples kernel
%    features: the embedded features used to construct the kernel
%    D: Distance matrix (only for Gaussian kernels)
%
% Christine Ahrends, Aarhus University, 2022

%% Load data

% load data (timecourses and pre-trained HMM)
load([datadir '/tc1001_restall.mat']) % data_X
load([hmmdir '/' HMM_name '.mat']) % HMM

%% construct kernels and feature matrices from HMM

% specify options for kernel construction:
% which HMM parameters to construct kernel from: 
%    Pi (initial state probabilities)
%    P (transition probabilities)
%    mu (state means)
%    sigma (state covariances)
% here: use all features but add PCA dimensionality reduction for state
% features
K_options = struct();
K_options.pca = true; % do PCA for state features
K_options.Pi = true;
K_options.P = true;
if HMM.hmm.train.zeromean==0
    K_options.mu = true; 
else
    K_options.mu = false;
end
K_options.sigma = true; 

% set which kernel to build:
types = {'Fisher', 'naive', 'naive_norm'};
shapes = {'linear', 'Gaussian'};

K_options.type = types{Fn}; % one of 'Fisher', 'naive', or 'naive_norm'
K_options.shape = shapes{Kn};

if ~isdir(kerneldir); mkdir(kerneldir); end

if Kn==2
    [Kernel, features, D] = hmm_kernel(data_X, HMM.hmm, K_options);
    save([kerneldir '/Kernel_PCAstates_' HMM_name '_' types{Fn} '_' shapes{Kn} '.mat'], 'Kernel', 'features', 'D');
else
    [Kernel, features] = hmm_kernel(data_X, HMM.hmm, K_options);
    save([kerneldir '/Kernel_PCAstates_' HMM_name '_' types{Fn} '_' shapes{Kn} '.mat'], 'Kernel', 'features');
end

end