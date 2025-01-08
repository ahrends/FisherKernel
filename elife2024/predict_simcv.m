function [Kernel, features, corr_test] = predict_simcv(datadir, hmmdir, outputdir, HMM_name, n_train, n_test, betwgroup_diff, Y_noise, cv)
% [Kernel, feat, corr_test] = predict_simcv(HMM_name, n_train, n_test, betwgroup_diff, Y_noise, cv)
%
% constructs kernels and predicts from HMMs run on simulated timecourses 
% for train and test sets with differing levels of noise. Uses the Fisher
% kernel and naive kernels in kernel ridge regression to predict simulated
% target variable. Assumes data was generated using simulate_cv_generatetc
% and HMMs were fit either only on the simulated training set or on all
% subjects using fit_HMM_cv.
% 
% Input:
%    HMM_name: root name for output HMMs to be recognised by kernel-builder
%       functions
%    n_train: number of subjects in the training set
%    n_test: number of subjects in the test set
%    betwgroup_diff: scalar for the difference between the training and 
%       test set
%    Y_noise: noise on the target variable
%    cv: whether to fit only to the training set or to all subjects
%       (1 for only training subjects, 2 for all subjects)
%
% Christine Ahrends, University of Oxford, 2024

%% Load example data

if cv==1
    cv_char = 'sep';
elseif cv==2
    cv_char = 'tog';
end

% load simulated data and trained HMM
load([datadir '/X_ntrain' num2str(n_train) '_ntest' num2str(n_test) ...
    '_betwgroupdiff' num2str(betwgroup_diff) '_Ynoise' num2str(Y_noise) '.mat']);
load([datadir '/Y_ntrain' num2str(n_train) '_ntest' num2str(n_test) ...
    '_betwgroupdiff' num2str(betwgroup_diff) '_Ynoise' num2str(Y_noise) '.mat']);
load([hmmdir '/' HMM_name '_' cv_char '_ntrain' num2str(n_train) '_ntest' num2str(n_test) ...
    '_betwgroupdiff' num2str(betwgroup_diff) '_Ynoise' num2str(Y_noise) '.mat']);

% get data for all subjects
n_subj = n_train+n_test;
Xc = cell(n_subj,1);
Tc = cell(n_subj,1);
for n = 1:n_subj
    Xc{n} = X(sum(T(1:n-1))+1:sum(T(1:n)),:);
    Tc{n} = T(n);
end

%% construct kernels and feature matrices from HMM and predict

% specify options for kernel construction:
% which HMM parameters to construct kernel from: 
%    Pi (initial state probabilities)
%    P (transition probabilities)
%    mu (state means)
%    sigma (state covariances)
% at the moment: use either a) Pi & P, b) Pi, P, and sigma, or c) Pi, P, mu, and sigma
K_options = struct();
K_options.Pi = true; % state probabilities
K_options.P = true; % transition probabilities
K_options.mu = true; % state means
K_options.sigma = true; % state covariances
K_options.shape = 'linear';
K_options.normalisation = [];
all_types = {'Fisher', 'naive', 'naive_norm'}; % use all available kernels

% split into training and test set
S = n_subj;
ind = 1:S;
train_ind = 1:n_train; % simulated data are ordered (training subjects first, then test subjects)
test_ind = ind(~ismember(ind, train_ind));

% initialise cells to hold features and kernels
features = cell(3,1);
Kernel = cell(3,1);

% predict Y
krr_options = struct();
krr_options.CVscheme = [1 2];
krr_options.CVfolds = {test_ind};
Yin = Y_true;
predictedY = cell(3,1);
corr_test = zeros(3,1); 
for kk = 1:3
    K_options.type = all_types{kk};
    [Kernel{kk},features{kk}] = hmm_kernel(Xc, HMM.hmm, K_options); % construct kernel (Fisher, naive, naive normalised)
    
    Din = Kernel{kk};
    [predictedY{kk},~,~,~] = predictPhenotype(Yin,Din,krr_options); % run kernel ridge regression
    corr_test(kk) = corr(Y_true(test_ind), predictedY{kk}(test_ind)); % correlation coefficient between true (simulated) Y and predicted Y in test set
    
end

save([outputdir '/Results_' HMM_name '_' cv_char '_ntrain' num2str(n_train) '_ntest' num2str(n_test) ...
    '_betwgroupdiff' num2str(betwgroup_diff) '_Ynoise' num2str(Y_noise) '.mat'], "Kernel", "features", "corr_test");

end

