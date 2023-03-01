%% Fisher Kernel: Main script

%% Preparation
% set directories and general variables, load data

projectdir = '/user/FisherKernel_project';
scriptdir = [projectdir '/scripts/FisherKernel']; % directory for this folder
hmm_scriptdir = [projectdir '/scripts/HMM-MAR-master'];
datadir = [projectdir '/data/HCP_1200']; % directory where HCP S1200 timecourses and behavioural/demographic variables can be found
outputdir = [projectdir '/results'];

cd(projectdir)
addpath(genpath(scriptdir))
addpath(genpath(hmm_scriptdir));

% load timecourses (here called 'data') that the HMM should be run on
load([datadir '/tcs/hcp1003_RESTall_LR_groupICA50.mat']);
% assuming here that timecourses are a subjects x 1 cell, each containing a
% timepoints x ROIs matrix

% load variables to be predicted
% should be a subjects x variables matrix
all_vars = load([datadir '/vars.txt']);
load([datadir '/headers_grouped_category.mat']) % headers of variables in all_vars
load([datadir '/vars_target_with_IDs.mat'])
int_vars = vars_target_with_IDs;
clear vars_target_with_IDs
target_ind = ismember(all_vars(:,1), int_vars(:,1)); % find indices of subjects for which we have int_vars

data_X = data(target_ind);
clear data all_vars headers_grouped_category int_vars

S = size(data_X,1); % number of subjects (for which at least one of the variables we want to predict are available)
N = size(data_X{1},2); % number of ROIs
T = cell(size(data_X,1),1); % cell containing no. of timepoints for each subject
for s = 1:S
    T{s} = size(data_X{s},1); % should be 4800 when using all four scanning sessions, 1200 when using only on
end

%% 1. Run group-level HMM

% Main text version: Gaussian HMM where states have mean and
% covariance run on all 4 scanning sessions per participant

K = 6; % number of HMM-states
hmm_options = struct();
hmm_options.order = 0; % Gaussian
hmm_options.covtype = 'full'; %('full' for covariance, 'uniquefull' for no covariance)
hmm_options.zeromean = 0; % (0 to model mean, 1 to model only covariance)
hmm_options.standardise = 1; 
hmm_options.dropstates = 0;
hmm_options.K = K;
% hmm_options.useParallel = 0; 
[HMM.hmm, HMM.Gamma, HMM.Xi, HMM.vpath, ~, ~, HMM.fehist] = hmmmar(data_X, T, hmm_options);
if ~isdir(outputdir); mkdir(outputdir); end
save([outputdir '/HMM_main.mat'], 'HMM')

% (Other versions for Supplementary Information)
% SI version 1: only first scanning session of each participant:
load([datadir '/tcs/hcp1003_REST1_LR_groupICA50.mat']);
data_X1 = data(target_ind);
T1 = cell(size(data_X1,1),1); % cell containing no. of timepoints for each subject
for s = 1:S
    T1{s} = size(data_X1{s},1); % should be 4800 when using all four scanning sessions, 1200 when using only on
end
clear HMM
[HMM.hmm, HMM.Gamma, HMM.Xi, HMM.vpath, ~, ~, HMM.fehist] = hmmmar(data_X1, T1, hmm_options);
save([outputdir '/HMM_rest1.mat'], 'HMM')

% SI version 2: Gaussian HMM where states have only covariance (mean set to
% 0):
hmm_options.zeromean = 1; % (0 to model mean, 1 to model only covariance)
clear HMM
[HMM.hmm, HMM.Gamma, HMM.Xi, HMM.vpath, ~, ~, HMM.fehist] = hmmmar(data_X, T, hmm_options);
save([outputdir '/HMM_cov.mat'], 'HMM')

%% 2. Construct kernels (and/or embedded features or distance/divergence matrices)

% (load HMM)
HMM_version = 'HMM_main'; % change to run on SI versions of HMM
load([outputdir '/' HMM_version '.mat'])

% construct kernels and feature matrices from HMM
% decide which parameters to use:
% Pi: state probabilities
% P: transition probabilities
% mu: state means
% sigma: state covariances
% at the moment: use either a) Pi & P, b) Pi, P, and sigma, or c) Pi, P, mu, and sigma
% we here used all available parameters, i.e. Pi, P, mu, and sigma for main
% text version and Pi, P, and sigma for HMM where states have only
% covariance
K_options = struct();
K_options.Pi = true;
K_options.P = true;
if HMM.hmm.train.zeromean==0
    K_options.mu = true; % use state means only if they were estimated
else
    K_options.mu = false; % otherwise use only state covariances
end
K_options.sigma = true;

features = {'naive', 'naive_norm', 'Fisher'};
kernels = {'linear', 'gaussian'};

for f=1:3
    for k=1:2
        K_options.type = features{f}; % one of 'naive', 'naive_norm', or 'Fisher'
        % 'naive' will also give the vectorised parameters, 'naive_norm' will also
        % give the normalised vectorised parameters, 'Fisher' will also give the
        % gradient features
        K_options.kernel = kernels{k};

        if ~isdir(outputdir); mkdir(outputdir); end
        clear Kernel features Dist
        if k==2
            [Kernel, features, Dist] = hmm_kernel(data_X, HMM.hmm, K_options);
            save([outputdir '/Kernel_' features{f} '_' kernels{k} '.mat'], 'Kernel', 'features', 'Dist');

        else
            [Kernel, features] = hmm_kernel(data_X, HMM.hmm, K_options);
            save([outputdir '/Kernel_' features{f} '_' kernels{k} '.mat'], 'Kernel', 'features');
        end
    end
end

% compare also to KL divergence model
clear Dist
Dist = computeDistMatrix(data_X, T, HMM.hmm);
save([outputdir '/Kernel_KLdiv.mat'], 'Dist');

% (SI: compare also to static FC KL divergence model)
clear Dist
Dist = computeDistMatrix_AVFC(data_X,T);
save([outputdir '/Kernel_KLdiv_staticFC.mat'], 'Dist')


