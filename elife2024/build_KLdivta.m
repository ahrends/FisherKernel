function build_KLdivta()

%% 
% build symmetrised KL divergence matrix based on time-averaged model (single-state HMM)
% The divergence matrix will be used in Gaussian kernel for KRR in main
% prediction function
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
load([datadir '/tc1001_restall.mat']) % data_X


%% compute divergence matrix

N = 1001; % number of subjects
T = cell(N,1); % make cell containing the number of timepoints for each session (same as when fitting the HMM, to define borders)
for n = 1:N
    T{n} = 4800;
end

D = computeDistMatrix_AVFC(data(target_ind),T);

save([outputdir '/Kernel_static_KLdiv.mat'], 'D')
end