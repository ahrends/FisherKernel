function HMM = fit_HMM(only_cov, k)

%%
% fit a group-level HMM to the HCP resting state fMRI data.
% This uses a Gaussian observation model, either using the mean or pinning
% it to 0 (only_cov).
% 
% Input:
%    only_cov: should be either 1 (to model states using only covariance) or 0
%    (to use both mean and covariance) 
%    k: number of HMM states
% 
% Output:
%    HMM: trained HMM struct containing
%       hmm: the estimated model (struct)
%       Gamma: the state probability timecourses (timepoints x k)
%       Xi: joint probability of past and future states conditioned data
%       (timepoints x k x k)
%       vpath: Viterbi path (timepoints x 1 vector)
%       fehist: Free Energy history to check model inference
%
% Christine Ahrends, Aarhus University, 2022

%% Preparation

% set directories
scriptdir = '/path/to/code';
hmm_scriptdir = '/path/to/HMM-MAR-master';
datadir = '/path/to/data';%[projectdir '/scratch/Kernel/data'];
outputdir = '/path/to/output';

if ~isdir(outputdir); mkdir(outputdir); end

addpath(scriptdir)
addpath(genpath(hmm_scriptdir))

%% load data (X: timecourses, Y: behavioural variable to be predicted)

% load X
load([datadir '/hcp1003_RESTall_LR_groupICA50.mat']);
% X are the timecourses that the HMM will be run on
% assuming here that timecourses are a subjects x 1 cell, each containing a
% timepoints x ROIs matrix

% load Y 
% Y are the variables to be predicted
% should be a subjects x variables matrix
% here only used to get indices for subjects where at least one of the
% behavioural variables is available

all_vars = load([datadir '/vars.txt']);
load([datadir '/headers_grouped_category.mat']) %headers of variables in all_vars
load([datadir '/vars_target_with_IDs.mat'])
int_vars = vars_target_with_IDs;
clear vars_target_with_IDs
target_ind = ismember(all_vars(:,1), int_vars(:,1)); % subject indices

data_X = data(target_ind); % remove subjects for which none of the behavioural variables are available
clear data all_vars headers_grouped_category int_vars

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

save([outputdir '/HMM_only_cov' num2str(only_cov) '.mat'], 'HMM')

end