function D = build_KLdivta(data, N, ts)

%% 
% build symmetrised KL divergence matrix based on time-averaged model (single-state HMM)
% The divergence matrix will be used in Gaussian kernel for KRR in main
% prediction function
% 
% Input:
%    data: timecourses
%    N: number of subjects
%    ts: number of timepoints per subject (assuming here that this will be
%    the same across subjects)
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
outputdir = '/path/to/kernels';

addpath(scriptdir)
addpath(genpath(hmm_scriptdir));

% load data (timecourses for 1001 subjects for which at least one
% behavioural variable is available


%% compute divergence matrix

T = cell(N,1); % make cell containing the number of timepoints for each session (same as when fitting the HMM, to define borders)
for n = 1:N
    T{n} = ts;
end

D = computeDistMatrix_AVFC(data,T);

save([outputdir '/Kernel_static_KLdiv.mat'], 'D')
end