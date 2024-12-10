function [Kernel, features, D] = build_kernels_main(HMM_name, only_cov, Fn, Kn)

%%
% build kernels from HMMs (wrapper for hmm_kernel.m)
%
% Input: 
%    Fn: select embedding: 1 for naive (no embedding), 2 for 
%        naive norm (normalised parameters across subjects), 3 for
%        gradient embedding (Fisher kernel)
%    Kn: select kernel shape: 1 for linear, 2 for Gaussian
% 
% Output:
%    Kernel: the subject x subject kernel
%    features: the embedded features used to construct the kernel
%    D: Distance matrix (only for Gaussian kernels)
%
% Christine Ahrends, Aarhus University, 2022

%% Preparation

% set directories
scriptdir = '/path/to/code';
hmm_scriptdir = '/path/to/HMM-MAR-master';
datadir = '/path/to/data'; % needs to contain timecourses for relevant subjects 
hmmdir = '/path/to/hmm'; % needs to contain pre-trained HMM
outputdir = '/path/to/kernels';

addpath(scriptdir)
addpath(genpath(hmm_scriptdir));

% load data (timecourses and pre-trained HMM)
load([datadir '/tc1001_restall.mat']) % data_X
load([hmmdir '/' HMM_name '_only_cov_' num2str(only_cov) '.mat'])

%% construct kernels and feature matrices from HMM

% specify options for kernel construction:
% which HMM parameters to construct kernel from: 
%    Pi (initial state probabilities)
%    P (transition probabilities)
%    mu (state means)
%    sigma (state covariances)
% at the moment: use either a) Pi & P, b) Pi, P, and sigma, or c) Pi, P, mu, and sigma
K_options = struct();
K_options.Pi = true;
K_options.P = true;
if HMM.hmm.train.zeromean==0
    K_options.mu = true;
else
    K_options.mu = false;
end
K_options.sigma = true;

% set which kernel to build:
types = {'naive', 'naive_norm', 'Fisher'};
shapes = {'linear', 'Gaussian'};

K_options.type = types{Fn}; % one of 'naive', 'naive_norm', or 'Fisher'
K_options.shape = shapes{Kn};

if ~isdir(outputdir); mkdir(outputdir); end

% build kernel and feature matrices
if Kn==2
    [Kernel, features, D] = hmm_kernel(data_X, HMM.hmm, K_options);
    save([outputdir '/Kernel_' HMM_name '_only_cov_' num2str(only_cov) '_' types{Fn} '_' shapes{Kn} '.mat'], 'Kernel', 'features', 'D');
    
else
    [Kernel, features] = hmm_kernel(data_X, HMM.hmm, K_options);
    save([outputdir '/Kernel_' HMM_name '_only_cov_' num2str(only_cov) '_' types{Fn} '_' shapes{Kn} '.mat'], 'Kernel', 'features');
end

end