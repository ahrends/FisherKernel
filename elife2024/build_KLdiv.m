function D = build_KLdiv(HMM_name, only_cov)
% D = build_KLdiv(HMM_name, only_cov)
%
% build symmetrised KL divergence matrix based on time-varying model (HMM)
% This requires that the group-level HMM has been fitted to the
% timecourses.
% The divergence matrix will be used in Gaussian kernel for KRR in main
% prediction function
%
% Input:
%    HMM_name: Name of the file containing the trained HMM
% 
% Output:
%    D: symmetrised KL divergence matrix (subjects x subjects)
%
% Christine Ahrends, University of Oxford, 2024

%% Preparation

% set directories
scriptdir = '/path/to/code';
hmm_scriptdir = '/path/to/HMM-MAR-master';
datadir = '/path/to/data';
hmmdir = '/path/to/hmm'; % needs to contain pre-trained group-level HMM
outputdir = '/path/to/kernels';

addpath(scriptdir)
addpath(genpath(hmm_scriptdir));

% load data (timecourses for 1001 subjects for which at least one
% behavioural variable is available
load([datadir '/tc1001_restall.mat']) % data_X
load([hmmdir '/' HMM_name '_only_cov_' num2str(only_cov) '.mat']) % HMM

%% compute divergence matrix

S = size(data_X,1); % number of subjects
T = cell(S,1); % make cell containing the number of timepoints for each session (same as when fitting the HMM, to define borders)
for s = 1:S
    T{s} = size(data_X{s},1); 
end

D = computeDistMatrix(data_X, T, HMM.hmm);

save([outputdir '/Kernel_' HMM_name '_only_cov_' num2str(only_cov) '_KLdiv.mat'], 'D');


end