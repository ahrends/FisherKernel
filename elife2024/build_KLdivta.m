function D = build_KLdivta(datadir, kerneldir, only_1st)
% D = build_KLdivta(datadir, kerneldir, only_1st)
%
% build symmetrised KL divergence matrix based on time-averaged model (single-state HMM)
% The divergence matrix will be used in Gaussian kernel for KRR in main
% prediction function
% 
% Dependencies:
% HMM-MAR toolbox: https://github.com/OHBA-analysis/HMM-MAR
% 
% Input:
%    datadir: directory for HCP rsFMRI timecourses
%    kerneldir: (output) directory where kernels and features will be saved
%    only_1st: whether to use only the first scanning session or all
% 
% Output:
%    D: symmetrised KL divergence matrix (subjects x subjects)
%
% Christine Ahrends, University of Oxford, 2024

%% Load data

% load timecourses
if only_1st==0
    load([datadir '/tc1001_restall.mat']) % data_X
elseif only_1st==1
    load([datadir '/tc1001_rest1.mat'])
end

%% compute divergence matrix

S = size(data_X,1); % number of subjects
T = cell(S,1); % make cell containing the number of timepoints for each session (same as when fitting the HMM, to define borders)
for s = 1:S
    T{s} = size(data_X{s},1); 
end

D = computeDistMatrix_AVFC(data_X, T); % symmetrised KL divergence matrix

if ~isdir(kerneldir); mkdir(kerneldir); end
save([kerneldir '/Kernel_static_KLdiv.mat'], 'D')

end