function HMM = fit_HMM(datadir, hmmdir, HMM_name, only_cov, only_1st, k)
% HMM = fit_HMM(datadir, hmmdir, HMM_name, only_cov, only_1st, k)
%
% fit a group-level HMM to the HCP resting state fMRI data.
% This uses a Gaussian observation model, either using the mean or pinning
% it to 0 (only_cov).
% Wrapper for hmmmar.
% 
% Dependencies:
% HMM-MAR toolbox: https://github.com/OHBA-analysis/HMM-MAR
% 
% Input:
%    datadir: directory where HCP rsfMRI timecourses are stored
%    hmmdir: (output) directory to save the HMM
%    HMM_name: root name for HMMs to be recognised by kernel-builder
%       functions
%    only_cov: should be either 1 (to model states using only covariance) or 0
%       (to use both mean and covariance) 
%    only_1st: use only 1st scanning session? (1 to use only 1st, 0 to use
%       all scanning sessions)
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
% Christine Ahrends, Aarhus University, 2022

%% Load data

% load timecourses
if only_1st==0
    load([datadir '/tc1001_restall.mat']); % data_X
elseif only_1st==1
    load([datadir '/tc1001_rest1.mat'])
end
% data_X are the timecourses that the HMM will be run on
% assuming here that timecourses are a subjects x 1 cell, each containing a
% timepoints x ROIs matrix

S = size(data_X,1);
N = size(data_X{1},2);
T = cell(size(data_X,1),1);
for s = 1:S
    T{s} = size(data_X{s},1);
end

%% Fit HMM:

% specify options
hmm_options = struct();
hmm_options.order = 0;
hmm_options.covtype = 'full'; %('full' for covariance, 'uniquefull' for no covariance)
hmm_options.zeromean = only_cov; % (0 to model mean, 1 to model only covariance)
hmm_options.standardise = 1; 
hmm_options.dropstates = 0;
hmm_options.K = k;
hmm_options.useParallel = 0;

% fit group-level HMM 
[HMM.hmm, HMM.Gamma, HMM.Xi, HMM.vpath, ~, ~, HMM.fehist] = hmmmar(data_X, T, hmm_options);

if ~isdir(hmmdir); mkdir(hmmdir); end
save([hmmdir '/' HMM_name '.mat'], 'HMM')

end