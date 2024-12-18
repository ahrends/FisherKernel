function HMM = fit_HMM_cv(datadir, kerneldir, hmmdir, HMM_name, leaveout_foldN, iterN, k)
% HMM = fit_HMM_cv(datadir, kerneldir, hmmdir, HMM_name, leaveout_foldN, iterN, k)
%
% fit a group-level HMM to the HCP resting state fMRI data.
% This uses a Gaussian observation model, either using the mean or pinning
% it to 0 (only_cov). This function loads pre-defined folds and fits the
% HMM to all but the specified fold (used later for testing the model).
% Wrapper for hmmmar.
% 
% Dependencies:
% HMM-MAR toolbox: https://github.com/OHBA-analysis/HMM-MAR
% 
% Input:
%    datadir: directory where HCP rsfMRI timecourses are stored
%    hmmdir: (output) directory to save the HMM
%    kerneldir: directory containing pre-defined folds
%    HMM_name: root name for HMMs to be recognised by kernel-builder
%       functions
%    leaveout_foldN: fold number used later for testing the model (leave
%       out during HMM training)
%    iterN: iteration number for folds (assuming folds were made using
%       make_folds)
%    k: number of HMM states
% 
% Output (will be written to hmmdir):
%    HMM: trained HMM struct containing
%       hmm: the estimated model (struct)
%       Gamma: the state probability timecourses (timepoints x k)
%       Xi: joint probability of past and future states conditioned data
%       (timepoints x k x k)
%       vpath: Viterbi path (timepoints x 1 vector)
%       fehist: Free Energy history to check model inference
%
% Christine Ahrends, University of Oxford, 2024

%% Load data

% load timecourses
load([datadir '/tc1001_restall.mat']); % data_X
% data_X are the timecourses that the HMM will be run on
% assuming here that timecourses are a subjects x 1 cell, each containing a
% timepoints x ROIs matrix

n_folds = 10;
load([kerneldir '/folds.mat']) % load pre-defined folds
train_folds = folds{iterN}(1:end ~= leaveout_foldN);
train_ind = [];
for i = 1:(n_folds-1)
    train_ind = [train_ind, train_folds{i}];
end

data_train = data_X(train_ind);

S = size(data_train,1);
N = size(data_train{1},2);
T = cell(size(data_train,1),1);
for s = 1:S
    T{s} = size(data_train{s},1);
end

%% Fit HMM:

% specify options
hmm_options = struct();
hmm_options.order = 0;
hmm_options.covtype = 'full'; %('full' for covariance, 'uniquefull' for no covariance)
hmm_options.zeromean = 0; % (0 to model mean, 1 to model only covariance)
hmm_options.standardise = 1; 
hmm_options.dropstates = 0;
hmm_options.K = k;
hmm_options.useParallel = 0;

% fit group-level HMM 
[HMM.hmm, HMM.Gamma, HMM.Xi, HMM.vpath, ~, ~, HMM.fehist] = hmmmar(data_train, T, hmm_options);

if ~isdir(hmmdir); mkdir(hmmdir); end
save([hmmdir '/' HMM_name '_leaveoutfoldN' num2str(leaveout_foldN) '_iterN' num2str(iterN) '.mat'], 'HMM')

end