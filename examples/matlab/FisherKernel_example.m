%% Fisher kernel: Example
% 
% Examples for constructing Fisher kernels from HMMs for use in prediction
% models
% 
% This script uses synthetic data we have generated for illustrative
% purposes and made available on OSF (Project page:
% https://osf.io/8qcyj/?view_only=119c38072a724e0091db5ba377935666). 
% Instead of using the synthetic data, you can of course use your own
% dataset. The required data format is explained in the relevant sections.
% Examples include different observation models (Gaussian HMM and
% time-delay embedded (TDE) HMM), and application examples for regression
% and classification problems (kernel ridge regression and support vector
% machines)
% 
% Christine Ahrends, Aarhus University (2022)

%% Add the HMM-MAR toolbox

hmm_codedir = '/path/to/HMM-MAR-master';
addpath(genpath(hmm_codedir));

%% Load example data from OSF

% load synthetic timecourses data_X that the HMM will be trained on, 
% here all subjects' timecourses are concatenated along first dimension,
% channels/regions are the second dimension
% But note that to construct the kernels, the data is expected to be in cell
% format rather than concatenated so we will reshape it below
url = 'https://osf.io/2b79h/download';
Xtmp = table2array(webread(url));
% load synthetic behavioural variable data_Y (should be a subjects x
% variables matrix) to be predicted
url = 'https://osf.io/ms8xa/download';
data_Y = table2array(webread(url));
% load confounds
url = 'https://osf.io/7phs4/download';
confounds = table2array(webread(url));
% load family structure (should be a subjects x subjects matrix)
url = 'https://osf.io/8pr27/download';
family = table2array(webread(url));

nvars = size(data_Y,2);
nsubs = size(data_Y,1); 
nts = size(Xtmp,1)/nsubs;

T = cell(nsubs,1); % indicates timepoints per subject/session
data_X = cell(nsubs,1);
for i = 1:nsubs
    T{i} = nts;
    data_X{i} = Xtmp((i-1)*nts+1:(i-1)*nts+nts,:);
end

%% Example 1: Gaussian HMM with mean and covariance

hmm_options1 = struct();
hmm_options1.order = 0;
hmm_options1.covtype = 'full'; %('full' for covariance, 'uniquefull' for no covariance)
hmm_options1.zeromean = 0; % (0 to model mean, 1 to model only covariance)
hmm_options1.standardise = 1; % important! standardise data when using kernel (will be passed down to hmm_kernel)
hmm_options1.dropstates = 0;
hmm_options1.K = 6;
hmm_options1.useParallel = 1; % parallelise if possible

% run HMM (group-level) - this might take a while especially when not
% using parallel option
[HMM1.hmm, HMM1.Gamma, HMM1.Xi, HMM1.vpath, ~, ~, HMM1.fehist] = hmmmar(data_X, T, hmm_options1);

%% Construct kernels and feature matrices from HMM

% at the moment: use either a) Pi & P, b) Pi, P, and sigma, or c) Pi, P, mu, and sigma
K_options = struct();
K_options.Pi = true; % state probabilities
K_options.P = true; % transition probabilities
K_options.mu = true; % state means
K_options.sigma = true; % state covariances
K_options.type = 'Fisher'; % one of 'naive', 'naive_norm', or 'Fisher'
K_options.shape = 'linear';
K_options.normalisation = 'L2-norm'; % only here for visualisation, drop this when running KRR

[FK, feat] = hmm_kernel(data_X, HMM1.hmm, K_options);
% FK is the Fisher kernel
figure; subplot(1,2,1); imagesc(feat); title('Gradient features (normalised)'); 
xlabel('Features'); ylabel('Subjects'); colorbar; 
subplot(1,2,2); imagesc(FK); title('Fisher Kernel (normalised)'); 
xlabel('Subjects'); ylabel('Subjects'); axis square; colorbar;

% alternatively, use Gaussian Fisher kernel:
K_options.shape = 'Gaussian';
K_options.tau = 100; % estimate tau in CV

[FKg, feat, Dist] = hmm_kernel(data_X, HMM1.hmm, K_options);
% FKg is the Gaussian version of the Fisher Kernel
% Dist is the distance matrix
figure; subplot(1,2,1); imagesc(feat); title('Gradient features (normalised)'); 
xlabel('Features'); ylabel('Subjects'); colorbar; 
subplot(1,2,2); imagesc(FKg); title('Gaussian Fisher Kernel (normalised)'); 
xlabel('Subjects'); ylabel('Subjects'); axis square; colorbar;

%% Example 2: TDE-HMM

hmm_options2 = struct();
hmm_options2.order = 0;
hmm_options2.covtype = 'full'; %('full' for covariance, 'uniquefull' for no covariance)
hmm_options2.standardise = 1; % important! standardise data when using kernel (will be passed down to hmm_kernel)
hmm_options2.dropstates = 0;
hmm_options2.K = 6;
hmm_options2.useParallel = 1;
% options specific for TDE-HMM:
hmm_options2.embeddedlags = -3:3;
hmm_options2.pca = 0.5; % just for illustration, this should be closer to 2*N_regions
hmm_options2.zeromean = 1; % needs to be 1 for TDE-HMM

% run HMM (group-level)
[HMM2.hmm, HMM2.Gamma, HMM2.Xi, HMM2.vpath, ~, ~, HMM2.fehist] = hmmmar(data_X, T, hmm_options2);

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
K_options.shape = 'linear';
K_options.normalisation = 'L2-norm'; % only here for visualisation, drop this when running KRR

[FK2, feat2] = hmm_kernel(data_X, HMM2.hmm, K_options);
% FK is the Fisher kernel
figure; subplot(1,2,1); imagesc(feat2); title('Gradient features (normalised)'); 
xlabel('Features'); ylabel('Subjects'); colorbar; 
subplot(1,2,2); imagesc(FK2); title('Fisher Kernel (normalised)'); 
xlabel('Subjects'); ylabel('Subjects'); axis square; colorbar;

% NOTE: gradient features will be in embedded space (e.g. embedded lags & 
% PCA space)

% alternatively, use Gaussian Fisher kernel:
K_options.shape = 'Gaussian';
K_options.tau = 100; % estimate tau in CV

[FKg2, feat2, Dist2] = hmm_kernel(data_X, HMM2.hmm, K_options);
% FKg is the Gaussian version of the Fisher Kernel
% Dist is the distance matrix
figure; subplot(1,2,1); imagesc(feat2); title('Gradient features (normalised)'); 
xlabel('Features'); ylabel('Subjects'); colorbar; 
subplot(1,2,2); imagesc(FKg2); title('Gaussian Fisher Kernel (normalised)'); 
xlabel('Subjects'); ylabel('Subjects'); axis square; colorbar;

%% Prediction example

% Here using kernel ridge regression, but the kernels constructed above can
% be used in any other kernel-based model
% This uses deconfounding and accounts for family structure

krr_params = struct();
krr_params.deconfounding = 1; % use deconfounding
krr_params.CVscheme = [10 10]; % number of outer and inner folds for nested CV
krr_params.alpha = [0.0001 0.001 0.01 0.1 0.3 0.5 0.7 0.9 1.0]; % ridge penalty (vector for grid search)
krr_params.verbose = 1;
krr_params.Nperm = 1; 
krr_params.shape = 'linear'; % either 'gaussian' or 'linear';

rng('shuffle')
% load predictors: 
% for linear kernel, load pre-constructed kernel itself, 
% for Gaussian kernel, load distance/divergence matrix and estimate
% tau (width of Gaussian kernel) within inner CV
if strcmpi(krr_params.shape, 'linear')
    Din = FK;
elseif strcmpi(krr_params.shape, 'gaussian')
    Din = Dist;
end

[predictedY, predictedYD, YD, stats] = predictPhenotype(data_Y, ...
    Din, krr_params, family, confounds);


%% Classification example

% download synthetic data for classification example:
clear Xtmp data_X data_Y
url = 'https://osf.io/nfg5v/download';
Xtmp = table2array(webread(url));
% the subjects in the simulated timeseries belong to two classes: the first
% 50 subjects are group 1 and the second 50 subjects are group 2
data_Y = [zeros(50,1); ones(50,1)];
data_Y = data_Y+1;

nsubs = size(data_Y,1); 
nts = size(Xtmp,1)/nsubs;

T = cell(nsubs,1); % indicates timepoints per subject/session
data_X = cell(nsubs,1);
for i = 1:nsubs
    T{i} = nts;
    data_X{i} = Xtmp((i-1)*nts+1:(i-1)*nts+nts,:);
end

% for efficiency, we are re-using the HMM fitted to the timeseries above
% (in reality, you should train the group-level HMM on the specific
% dataset)
K_options.normalisation = []; 
K_options.tau = [];
[FK, feat] = hmm_kernel(data_X, HMM1.hmm, K_options);

% split into training and test set
rng('shuffle')
ind = 1:nsubs;
split = 0.8; % ratio to split data into training and test set (variable is training set percentage)
train_size = round(nsubs*split);
test_size = nsubs-train_size;
[train_ind] = randsample(ind, train_size);
%size(unique(train_ind)) % check that the sampling picked (split) percentage of unique values from the dataset
test_ind = ind(~ismember(ind, train_ind));
train_feat = feat(train_ind,:);
test_feat = feat(test_ind,:);
Y_train = data_Y(train_ind,1); %binary variable
Y_test = data_Y(test_ind,1);

% fit SVM to predict binary variable in test set from gradient features
svm1 = fitcsvm(train_feat, Y_train, 'Standardize', false, 'KernelFunction', 'linear'); % linear kernel on the gradient features will essentially give the practical Fisher kernel
[est_labels, score] = predict(svm1, test_feat);
err1 = sum(est_labels ~= Y_test) ./numel(Y_test);
disp(['Generalization error on test set (Fisher kernel): ' num2str(err1)]);
