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

% load Y: variables to be predicted and confounds
% should be a subjects x variables matrix
all_vars = load([datadir '/vars.txt']);
load([datadir '/headers_grouped_category.mat']) % headers of variables in all_vars
load([datadir '/vars_target_with_IDs.mat'])
int_vars = vars_target_with_IDs;
clear vars_target_with_IDs
target_ind = ismember(all_vars(:,1), int_vars(:,1)); % find indices of subjects for which we have int_vars
confounds = all_vars(target_ind,[3,8]);
% create family structure to use for CV folds (produces variable "twins")
make_HCPfamilystructure;
% concatenate variables to be predicted (here: age and 34 intelligence
% variables)
Y = [pred_age(target_ind),int_vars(:,2:end)];

% load X: timecourses (here called 'data') that the HMM should be run on
load([datadir '/tcs/hcp1003_RESTall_LR_groupICA50.mat']);
% assuming here that timecourses are a subjects x 1 cell, each containing a
% timepoints x ROIs matrix

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

Fnames = {'naive', 'naive_norm', 'Fisher'};
Knames = {'linear', 'gaussian'};

% This will construct the kernels and the features (compute the gradient
% for Fisher kernel) and run dual estimation (Fisher-compatible version)
for f=1:3
    for k=1:2
        K_options.type = Fnames{f}; % one of 'naive', 'naive_norm', or 'Fisher'
        % 'naive' will also give the vectorised parameters, 'naive_norm' will also
        % give the normalised vectorised parameters, 'Fisher' will also give the
        % gradient features
        K_options.kernel = Knames{k};

        if ~isdir(outputdir); mkdir(outputdir); end
        clear Kernel features Dist
        if k==2
            [Kernel, features, Dist] = hmm_kernel(data_X, HMM.hmm, K_options);
            save([outputdir '/Kernel_' Fnames{f} '_' Knames{k} '.mat'], 'Kernel', 'features', 'Dist');
        else
            [Kernel, features] = hmm_kernel(data_X, HMM.hmm, K_options);
            save([outputdir '/Kernel_' Fnames{f} '_' Knames{k} '.mat'], 'Kernel', 'features');
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

%% 3. Run KRR for prediction
% Do not run
% set up variables & options for KRR
N_variables = 35;
N_iter = 100;

krr_params = struct();
krr_params.deconfounding = 1;
krr_params.CVscheme = [10 10];
krr_params.alpha = [0.0001 0.001 0.01 0.1 0.3 0.5 0.7 0.9 1.0];
krr_params.verbose = 1;
krr_params.Nperm = 1; % 100 (for permutation-based significance testing)

for f=1:3
    for k=1:2
        for varN = 1:N_variables
            for iterN = 1:N_iter
                if ~exist([outputdir '/Predictions_' HMM_version '_' Fnames{f} '_' Knames{k} '_varN' num2str(varN) 'iterN' num2str(iterN) '.mat'], 'file')
                    % load kernel (for linear kernel) or
                    % distance/divergence matrix (for Gaussian kernel)
                    if strcmpi(Fnames{f}, 'KL')
                        load([outputdir '/Kernel_KLdiv.mat'], 'Dist'); % load KL divergence matrix
                    else
                        if strcmpi(Knames{k}, 'linear')
                            load([outputdir '/Kernel_' Fnames{f} '_' Knames{k} '.mat'], 'Kernel');
                        elseif strcmpi(kernel, 'gaussian')
                            load([outputdir '/Kernel_' Fnames{f} '_' Knames{k} '.mat'], 'Dist');
                        end
                    end
                    krr_params.kernel = Knames{k}; %'gaussian','linear';
                    results = struct();
                    Yin = Y(:,varN);
                    index = ~isnan(Yin);
                    Yin = Yin(index,:);
                    rng('shuffle')
                    % choose input: use Kernel for linear kernel and distance matrix for
                    % Gaussian kernel (to estimate kernel width within KRR CV)
                    if strcmpi(Knames{k}, 'linear')
                        Din = Kernel(index,index);
                    elseif strcmpi(Knames{k}, 'gaussian')
                        Din = Dist(index, index);
                    end
                    % disp(['Now running iteration ' num2str(i) ' out of ' num2str(niter) ...
                    %    ' for variable ' num2str(j) ' out of ' num2str(nvars) ' for ' ...
                    %       Knames{k} ' ' Fnames{f} ' kernel']);
                    [results.predictedY, results.predictedYD, results.YD, results.stats] = predictPhenotype_kernels(Yin, ...
                        Din, krr_params, twins(index,index), confounds(index,:)); % twins is the family structure used for CV
    
                    if ~isdir(outputdir); mkdir(outputdir); end
                    save([outputdir '/Predictions_' HMM_version '_' Fnames{f} '_' Knames{k} '_varN' num2str(varN) 'iterN' num2str(iterN) '.mat'], 'results');
                end
            end
        end
    end
end

% (SI: run also for static FC KL divergence)
load([outputdir '/Kernel_KLdiv_staticFC.mat'], 'Dist');
krr_params.kernel = 'Gaussian';
Din = Dist(index, index);
for varN = 1:N_variables
    for iterN = 1:N_iter
        results = struct();
        Yin = Y(:,varN);
        index = ~isnan(Yin);
        Yin = Yin(index,:);
        rng('shuffle')
        % disp(['Now running iteration ' num2str(i) ' out of ' num2str(niter) ...
        %    ' for variable ' num2str(j) ' out of ' num2str(nvars) ' for static FC KL']);
        [results.predictedY, results.predictedYD, results.YD, results.stats] = predictPhenotype_kernels(Yin, ...
            Din, krr_params, twins(index,index), confounds(index,:));
        if ~isdir(outputdir); mkdir(outputdir); end
        save([outputdir '/Predictions_' HMM_version '_staticFCKL_varN' num2str(varN) 'iterN' num2str(iterN) '.mat'], 'results');
    end
end

