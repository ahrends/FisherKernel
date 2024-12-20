%% Fisher Kernel: elife 2024 Main script
% Ahrends, Woolrich, & Vidaurre (2024). Predicting individual traits from
% models of brain dynamics accurately and reliably using the Fisher kernel.
% elife https://doi.org/10.7554/eLife.95125.

% This script goes through the steps to reproduce the main results of the
% paper. For the simulations-based results, see SimulateFeatures_main.m and
% SimulateCV_main.m.
% 
% Dependencies:
% HMM-MAR toolbox - https://github.com/OHBA-analysis/HMM-MAR
% NetsPredict - https://github.com/vidaurre/NetsPredict
% covariancetoolbox - https://github.com/alexandrebarachant/covariancetoolbox
%
% Christine Ahrends, University of Oxford 2024

%% NOTE
% All functions are intended for a computing cluster. It is not 
% recommended to run this entire script without an appropriate computing 
% environment. All output will be written to the specified directories 
% rather than loading it into memory. If you are looking for an example or 
% guidance on how to use the Fisher kernel or predict from HMMs in your 
% own data, see the examples in this repository under 
% FisherKernel/examples/matlab and FisherKernel/examples/python for the 
% equivalent Python notebooks instead.

%% 0. Preparation
% set directories and general variables, load data

codedir = '/path/to/FisherKernel'; % directory for this folder
hmm_codedir = '/path/to/HMM-MAR-master'; % directory for HMM-MAR toolbox 
netspredictdir = '/path/to/NetsPredict-master'; % directory for NetsPredict toolbox
covariancedir = '/path/to/covariance-toolbox'; % directory for covariance-toolbox
datadir = '/path/to/data'; % directory where HCP S1200 rsfMRI timecourses, behavioural/demographic variables, and family structure can be found
hmmdir = '/path/to/hmm'; % directory where HMMs will be saved
kerneldir = '/path/to/kernels'; % directory where kernels, features, & folds will be saved
resultsdir = '/path/to/results';

addpath(genpath(codedir))
addpath(genpath(hmm_codedir))
addpath(genpath(netspredictdir))
addpath(genpath(covariancedir))

%% 0a. Check data availability
% load Y: variables to be predicted and confounds
% should be a subjects x variables matrix
all_vars = load([datadir '/vars.txt']);
load([datadir '/headers_grouped_category.mat']) % headers of variables in all_vars
load([datadir '/vars_target_with_IDs.mat'])
age = all_vars(:,4);
int_vars = vars_target_with_IDs;
clear vars_target_with_IDs
target_ind = ismember(all_vars(:,1), int_vars(:,1)); % find indices of subjects for which we have int_vars
confounds = all_vars(target_ind,[3,8]);
% create family structure to use for CV folds (produces variable "twins")
make_HCPfamilystructure;
% concatenate variables to be predicted (here: age and 34 intelligence
% variables)
Y = [age(target_ind),int_vars(:,2:end)];

% load X: timecourses (here called 'data') that the HMM should be run on
load([datadir '/tcs/hcp1003_RESTall_LR_groupICA50.mat']);
% assuming here that timecourses are a subjects x 1 cell, each containing a
% timepoints x ROIs matrix

data_X = data(target_ind);
clear data all_vars headers_grouped_category int_vars age
save([datadir '/tc1001_restall.mat'], "data_X");

S = size(data_X,1); % number of subjects (for which at least one of the variables we want to predict are available)
N = size(data_X{1},2); % number of ROIs
T = cell(size(data_X,1),1); % cell containing no. of timepoints for each subject
for s = 1:S
    T{s} = size(data_X{s},1); % should be 4800 when using all four scanning sessions, 1200 when using only one
end

% (Other versions for Supplementary Information)
% SI version 1: only first scanning session of each participant:
clear data_X T
load([datadir '/tcs/hcp1003_REST1_LR_groupICA50.mat']);
data_X = data(target_ind);
T = cell(size(data_X,1),1); % cell containing no. of timepoints for each subject
for s = 1:S
    T{s} = size(data_X{s},1); % should be 4800 when using all four scanning sessions, 1200 when using only one
end
save([datadir '/tc1001_rest1.mat'], "data_X");
clear data_X T

%% 1. Fit group-level HMM

% Main text version: Gaussian HMM where states have mean and
% covariance run on all 4 scanning sessions per participant
HMM_name = 'HMM_main'; % file name for this HMM to be passed on to kernel builders
k = 6; % number of states
only_cov = 0; % whether to use only covariance (and pin the mean to zero)
only_1st = 0; % whether to use only the first scanning session
fit_HMM(datadir, hmmdir, HMM_name, only_cov, only_1st, k);

% (Other versions for Supplementary Information)
% SI version 1: only first scanning session of each participant:
clear HMM
HMM_name = 'HMM_SI_only1st'; % file name for this HMM to be passed on to kernel builders
only_1st = 1; % whether to use only the first scanning session
fit_HMM(datadir, hmmdir, HMM_name, only_cov, only_1st, k);

% SI version 2: Gaussian HMM where states have only covariance (mean set to
% 0):
clear HMM
HMM_name = 'HMM_SI_onlycov';
only_1st = 0;
only_cov = 1;
fit_HMM(datadir, hmmdir, HMM_name, only_cov, only_1st,k);

%% 2. Construct kernels, distance/divergence matrices, and feature matrices

HMM_name = 'HMM_main'; % change to run on SI versions of HMM: % 'HMM_SI_only1st'; 'HMM_SI_onlycov';
only_1st = 0;

types = {'Fisher', 'naive', 'naive_norm'};
shapes = {'linear', 'Gaussian'};
verbose = 1;

% build and save all combinations of main kernels
for i = 1:numel(types)
    for j = 1:numel(shapes)
        if verbose
            disp(['Now building ' shapes{j} ' ' types{i} ' kernel'])
        end
        build_kernels_main(datadir, hmmdir, kerneldir, HMM_name, only_1st, i, j);
    end
end

% build and save HMM-based (time-varying) KL divergence matrix
if verbose; disp('Now building KL divergence matrix'); end
build_KLdiv(datadir, hmmdir, kerneldir, HMM_name, only_1st); 

% Time-averaged methods:

% build and save time-averaged KL divergence matrix
if verbose; disp('Now building static KL divergence matrix'); end
build_KLdivta(datadir, kerneldir, only_1st);

% for all other time-averaged methods, construct and save subject
% time-averaged covariance matrices:
if verbose; disp('Now computing static covariance matrices'); end
n_sess = 4; % to use all sessions (session borders will be taken into account); change to 1 to use only 1st session
make_staticFCcov(datadir, n_sess);

% build and save log-Euclidean distance matrix
if verbose; disp('Now building log-Euclidean distance matrix'); end
build_logEuclidean(datadir, kerneldir);

% all other methods use the static FC covariance matrices as input

%% 3. Predict individual traits
% For kernelised regression, use predict_kernels.m
% For non-kernel regression from static FC features, use predict_static.m,
% except for Selected Edges method, which is predict_selectededges.m

HMM_name = 'HMM_main'; % change to run on SI versions
% make folds for reliability assessment:
n_reps = 100; % 100 repetitions of
n_folds = 10; % 10-fold CV
make_folds(datadir, kerneldir, n_reps, n_folds);
% folds will be saved in kerneldir and loaded by prediction functions to
% ensure that the same combinations of subjects are used by all methods

n_vars = 35; % number of behavioural variables/phenotypes to be predicted

types = {'Fisher', 'naive', 'naive normalised', 'KL divergence', 'static KL divergence', 'log-Euclidean'};
shapes = {'linear', 'Gaussian'};
verbose = 1;
% main HMM-based kernels (Fisher, naive, naive norm.)
for Fn = 1:3 
    for Kn = 1:2 % do both linear and Gaussian versions
        for varN = 1:n_vars
            for iterN = 1:n_reps
                if verbose
                    disp(['Now running KRR with ' shapes{Kn} ' ' types{Fn} ' kernel, for variable #' ...
                        num2str(varN) ' out of ' num2str(n_vars) ' and iteration #' num2str(iterN) ...
                        ' out of ' num2str(n_reps)]);
                end
                predict_kernels(datadir, kerneldir, resultsdir, HMM_name, varN, iterN, Fn, Kn);
            end
        end
    end
end

% KL divergence (time-varying & time-averaged) & log-Euclidean
for Fn = 4:6
    Kn = 2; % only Gaussian kernel
    for varN = 1:n_vars
        for iterN = 1:n_reps
            if verbose
                disp(['Now running KRR with ' shapes{Kn} ' ' types{Fn} ' kernel, for variable #' ...
                    num2str(varN) ' out of ' num2str(n_vars) ' and iteration #' num2str(iterN) ...
                    ' out of ' num2str(n_reps)]);
            end
            predict_kernels(datadir, kerneldir, resultsdir, HMM_name, varN, iterN, Fn, Kn);
        end
    end
end

% static FC non-kernel methods
penalties = {'Ridge', 'Elastic Net'};
spaces = {'Euclidean', 'Riemannian'};

for EN = 0:1
    for Riem = 0:1
        for varN = 1:n_vars
            for iterN = 1:n_reps
                if verbose
                    disp(['Now running ' penalties{EN+1} ' regression in ' spaces{Riem+1} ' space, for variable #' ...
                        num2str(varN) ' out of ' num2str(n_vars) ' and iteration #' num2str(iterN) ...
                        ' out of ' num2str(n_reps)]);
                end
                predict_static(datadir, kerneldir, resultsdir, varN, iterN, EN, Riem);
            end
        end
    end
end

% Selected Edges method (Rosenberg et al. 2018 & Shen et al. 2018)
for varN = 1:n_vars
    for iterN = 1:n_reps
        if verbose
            disp(['Now running Selected Edges, for variable #' num2str(varN) ...
                ' out of ' num2str(n_vars) ' and iteration #' num2str(iterN) ...
                ' out of ' num2str(n_reps)]);
        end
        predict_selectededges(datadir, kerneldir, resultsdir, varN, iterN);
    end
end

%% 4. assemble results
% load all predictions and outcome measures, and assemble everything into a table

results_options = struct();
results_options.main = true;

collect_results(resultsdir, options); % this will write a table called MAINresultsT.csv into the results directory
% export to do stats & figures in R

%% Effect of feature subsets (Results 2.3)
% Here only running jobs in real data, see SimulateFeatures_main.m for
% simulation results
% Folds are the same as the ones generated above

HMM_name = 'HMM_main';
types = {'Fisher', 'naive', 'naive normalised'};

verbose = 1;

% build kernels excluding state features:
for Fn = 1:numel(types)
    if verbose
        disp(['Now building ' types{Fn} ' kernel excluding state features'])
    end
    build_kernels_nostates(datadir, hmmdir, kerneldir, HMM_name, Fn, 1);
end

% build kernels excluding transition features:
for Fn = 1:numel(types)
    if verbose
        disp(['Now building ' types{Fn} ' kernel excluding transition features'])
    end
    build_kernels_noPiP(datadir, hmmdir, kerneldir, HMM_name, Fn, 1);
end

% build kernels with PCA-reduced state features:
for Fn = 1:numel(types)
    if verbose
        disp(['Now building ' types{Fn} ' kernel with PCA-reduced state features'])
    end
    build_kernels_PCAstates(datadir, hmmdir, kerneldir, HMM_name, Fn, 1);
end

% predict from kernels with different feature subsets
featsets = {'no transition features', 'no states', 'PCA-reduced states'};

for Fn = 1:numel(types) % all kernels
    for FSn = 1:numel(featsets) % all feature subsets
        for varN = 1:n_vars % all target variables
            for iterN = 1:n_reps % for 100 iterations
                if verbose
                    disp(['Now running KRR with ' featsets{FSn} ' ' types{Fn} ' kernel, for variable #' ...
                        num2str(varN) ' out of ' num2str(n_vars) ' and iteration #' num2str(iterN) ...
                        ' out of ' num2str(n_reps)]);
                end
                predict_featsets(datadir, kerneldir, resultsdir, HMM_name, varN, iterN, Fn, 1, FSn);
            end
        end
    end
end

% assemble results
results_options = struct();
results_options.featuresets = true;

collect_results(resultsdir, results_options); % this will write a file called FEATSETSresultsT.csv to resultsdir

%% Effect of HMM training scheme (Results 2.4)
% Here only running jobs in real data, see SimulateCV_main.m for simulation
% results

% use first iteration of folds generated above to set training and test set

% fit HMM only to training sets (this will fit 10 separate HMMs):
HMM_name = 'HMM_cv';
n_reps = 1; % do this only once
n_folds = 10; % 10 folds
k = 6;
verbose = 1;

for ii = 1:n_folds % leaving out one fold at a time from fitting HMM
    if verbose
        disp(['Now fitting HMM leaving out fold #' num2str(ii)]);
    end
    fit_HMM_cv(datadir, kerneldir, hmmdir, HMM_name, ii, 1, k); % only 1 iteration
end

% build kernels from HMMs fit only to training subjects
types = {'Fisher', 'naive', 'naive normalised'};
shapes = {'linear', 'Gaussian'};

for Fn = 1:numel(types)
    for Kn = 1:numel(shapes)
        for ii = 1:n_folds
            if verbose
                disp(['Now building ' shapes{Kn} ' ' types{Fn} ' kernel for HMM trained leaving out fold #' ...
                    num2str(ii) ' out of ' num2str(n_folds)]);
            end
            build_kernels_cv(datadir, hmmdir, kerneldir, HMM_name, ii, Fn, Kn, 1); % only 1 iteration
        end
    end
end

% predict individual traits from kernels constructed from HMMs fitted with
% held-out folds
training_schemes = {'together', 'separate'};
iterN = 1; % only running 1 iteration here

for CVn = 0:1
    for Fn = 1:numel(types)
        for Kn = 1:numel(shapes)
            for varN = 1:n_vars
                if verbose
                    disp(['Now running KRR with ' shapes{Kn} ' ' types{Fn} ' kernel using HMM trained '
                        training_schemes{CVn+1} ', for variable #' ...
                        num2str(varN) ' out of ' num2str(n_vars) ' and iteration #' num2str(iterN) ...
                        ' out of ' num2str(n_reps)]);
                end
                predict_cv(datadir, kerneldir, resultsdir, HMM_name, varN, iterN, Fn, Kn, CVn);
            end
        end
    end
end

% assemble results
results_options = struct();
results_options.CV = true;

collect_results(resultsdir, results_options); % this will write a file called CVresultsT.csv to resultsdir

%% Statistics and figures
% All statistical comparisons and figures are done in R and can be found 
% in StatsFigures.R