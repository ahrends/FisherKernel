%% Fisher kernel: Example
% 
% Examples for constructing Fisher kernel from HMM for use in prediction
% models
% Example 1: Gaussian HMM
% Example 2: TDE-HMM
% Prediction model examples: SVM, KRR
% 
% Christine Ahrends, Aarhus University (2022)

%% set paths for data, scripts and dependencies
projectdir = '/projects/FisherKernel';
scriptdir = [projectdir '/scripts/FisherKernel'];
hmm_scriptdir = [projectdir '/scripts/HMM-MAR-master'];
datadir = [projectdir '/scratch/Kernel/data'];

cd(projectdir)
addpath(genpath(scriptdir))
addpath(genpath(hmm_scriptdir));

%% load data (X: timecourses, Y: behavioural variable to be predicted)

% load X
load([datadir '/example_data.mat'], 'X');
% X are the timecourses that the HMM will be run on
% assuming here that timecourses are a subjects x 1 cell, each containing a
% timepoints x ROIs matrix

% load Y 
load([datadir '/example_data.mat'], 'Y');
nvars = size(Y,2);
% Y are the variables to be predicted
% should be a subjects x variables matrix

S = size(X,1);
N = size(X{1},2);
T = cell(S,1);
for s = 1:S
    T{s} = size(X{s},1);
end

%% Example 1: Gaussian HMM with mean and covariance

hmm_options1 = struct();
hmm_options1.order = 0;
hmm_options1.covtype = 'full'; %('full' for covariance, 'uniquefull' for no covariance)
hmm_options1.zeromean = 0; % (0 to model mean, 1 to model only covariance)
hmm_options1.standardise = 1; % important! standardise data when using kernel (will be passed down to hmm_kernel)
hmm_options1.dropstates = 0;
hmm_options1.K = 6;
hmm_options1.useParallel = 0;

% run HMM (group-level)
[HMM1.hmm, HMM1.Gamma, HMM1.Xi, HMM1.vpath, ~, ~, HMM1.fehist] = hmmmar(X, T, hmm_options1);

%% construct kernels and feature matrices from HMM

% at the moment: use either a) Pi & P, b) Pi, P, and sigma, or c) Pi, P, mu, and sigma
K_options = struct();
K_options.Pi = true; % state probabilities
K_options.P = true; % transition probabilities
K_options.mu = true; % state means
K_options.sigma = true; % state covariances
K_options.type = 'Fisher'; % one of 'naive', 'naive_norm', or 'Fisher'
K_options.kernel = 'linear';
K_options.normalisation = 'L2-norm'; % only here for visualisation, drop this when running KRR

[FK, feat] = hmm_kernel(X, HMM1.hmm, K_options);
% FK is the Fisher kernel
% figure; subplot(1,2,1); imagesc(feat); title('Gradient features (normalised)'); 
% xlabel('Features'); ylabel('Subjects'); colorbar; 
% subplot(1,2,2); imagesc(FK); title('Fisher Kernel (normalised)'); 
% xlabel('Subjects'); ylabel('Subjects'); axis square; colorbar;

% alternatively, use Gaussian Fisher kernel:
K_options.kernel = 'Gaussian';
K_options.tau = 100; % estimate tau in CV

[FKg, feat, Dist] = hmm_kernel(X, HMM1.hmm, K_options);
% FKg is the Gaussian version of the Fisher Kernel
% Dist is the distance matrix
% figure; subplot(1,2,1); imagesc(feat); title('Gradient features (normalised)'); 
% xlabel('Features'); ylabel('Subjects'); colorbar; 
% subplot(1,2,2); imagesc(FKg); title('Gaussian Fisher Kernel (normalised)'); 
% xlabel('Subjects'); ylabel('Subjects'); axis square; colorbar;

%% Example 2: TDE-HMM

hmm_options2 = struct();
hmm_options2.order = 0;
hmm_options2.covtype = 'full'; %('full' for covariance, 'uniquefull' for no covariance)
hmm_options2.standardise = 1; % important! standardise data when using kernel (will be passed down to hmm_kernel)
hmm_options2.dropstates = 0;
hmm_options2.K = 6;
hmm_options2.useParallel = 0;
% options specific for TDE-HMM:
hmm_options2.embeddedlags = -7:7;
hmm_options2.pca = 0.5; % just for illustration, this should be closer to 2*N_regions
hmm_options2.zeromean = 1; % needs to be 1 for TDE-HMM

% run HMM (group-level)
[HMM2.hmm, HMM2.Gamma, HMM2.Xi, HMM2.vpath, ~, ~, HMM2.fehist] = hmmmar(X, T, hmm_options2);

% gradient works in embedded space, check transformed data:
% [Xt_tmp, Tt] = transform_dataHMM(X,T, hmm_options2);
% 
%% construct kernels and feature matrices from HMM

% at the moment: use either a) Pi & P, b) Pi, P, and sigma, or c) Pi, P, mu, and sigma
K_options = struct();
K_options.Pi = true; % state probabilities
K_options.P = true; % transition probabilities
K_options.mu = false; % state means
K_options.sigma = true; % state covariances
K_options.type = 'Fisher'; % one of 'naive', 'naive_norm', or 'Fisher'
K_options.kernel = 'linear';
K_options.normalisation = 'L2-norm'; % only here for visualisation, drop this when running KRR

[FK2, feat2] = hmm_kernel(X, HMM2.hmm, K_options);
% FK is the Fisher kernel
% figure; subplot(1,2,1); imagesc(feat2); title('Gradient features (normalised)'); 
% xlabel('Features'); ylabel('Subjects'); colorbar; 
% subplot(1,2,2); imagesc(FK2); title('Fisher Kernel (normalised)'); 
% xlabel('Subjects'); ylabel('Subjects'); axis square; colorbar;

% NOTE: gradient features will be in embedded space (e.g. embedded lags & 
% PCA space)

% alternatively, use Gaussian Fisher kernel:
K_options.kernel = 'Gaussian';
K_options.tau = 100; % estimate tau in CV

[FKg2, feat2, Dist2] = hmm_kernel(X, HMM2.hmm, K_options);
% FKg is the Gaussian version of the Fisher Kernel
% Dist is the distance matrix
% figure; subplot(1,2,1); imagesc(feat2); title('Gradient features (normalised)'); 
% xlabel('Features'); ylabel('Subjects'); colorbar; 
% subplot(1,2,2); imagesc(FKg2); title('Gaussian Fisher Kernel (normalised)'); 
% xlabel('Subjects'); ylabel('Subjects'); axis square; colorbar;


%% Prediction example 1: SVM

K_options.normalisation = []; 
K_options.tau = [];
[FK, feat] = hmm_kernel(X, HMM1.hmm, K_options);

% split into training and test set
rng('shuffle')
ind = 1:S;
split = 0.8; % ratio to split data into training and test set (variable is training set percentage)
train_size = round(S*split);
test_size = S-train_size;
[train_ind] = randsample(ind, train_size);
%size(unique(train_ind)) % check that the sampling picked (split) percentage of unique values from the dataset
test_ind = ind(~ismember(ind, train_ind));
train_feat = feat(train_ind,:);
test_feat = feat(test_ind,:);
Y_train = Y(train_ind,1); %binary variable
Y_test = Y(test_ind,1);

% fit SVM to predict binary variable in test set from gradient features
svm1 = fitcsvm(train_feat, Y_train, 'Standardize', false, 'KernelFunction', 'linear'); % linear kernel on the gradient features will essentially give the practical Fisher kernel
[est_labels, score] = predict(svm1, test_feat);
err1 = sum(est_labels ~= Y_test) ./numel(Y_test);
disp(['Generalization error on test set (Fisher kernel): ' num2str(err1)]);

%% Prediction example 2: kernel ridge regression
% set parameters for KRR
% niter = 1;
krr_options = struct();
krr_options.deconfounding = 0;
krr_options.CVscheme = [10 10];
% krr_params.alpha = [0.0001 0.001 0.01 0.1 0.3 0.5 0.7 0.9 1.0];
krr_options.verbose = 1;
%krr_params.sigmafact = 1;
krr_options.Nperm = 1; % (only relevant for permutation-based significance testing)
krr_options.kernel = 'linear'; % krr_predict_FK uses feature matrices as input and is inefficient for Gaussian kernel (use krr_predict with distance matrices as input instead)

varN = 2;
Yin = Y(:,varN);
index = ~isnan(Yin);
Yin = Yin(index,:);
Xin = feat(index,:);

%disp(['Now running iteration ' num2str(i) ' out of ' num2str(niter) ...
%    ' for variable ' num2str(j) ' out of ' num2str(nvars) ' for FK']);
[results.predictedY, results.stats] = ...
    krr_predict_FK(Yin,Xin, krr_options, twins(index,index)); %twins is optional family structure (CV set up to randomise folds taking family structure into account)
disp(['Prediction accuracy (Fisher kernel) is ' num2str(results.stats.corr)])

% use predictPhenotype_kernels instead of krr_predict_FK for deconfounding