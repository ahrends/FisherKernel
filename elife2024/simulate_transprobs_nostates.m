function [X, HMM, features, Kernel, err] = simulate_transprobs_nostates(HMM_name, n_subj, betwgroup_diff)
% [X, HMM, features, Kernel, err] = simulate_transprobs_nostates(HMM_name, n_subj, betwgroup_diff)
%
% simulate timecourses for two groups where the group difference lies in
% the transition probabilities, then construct kernels (**excluding** state
% parameters), classify, and calculate error in recovering the ground truth
%
% Dependencies:
% HMM-MAR toolbox: https://github.com/OHBA-analysis/HMM-MAR
%
% Input: 
%    HMM_name: root file name of the HMM to be loaded (this will be the
%       basis for generating synthetic timecourses)
%    n_subj: number of subjects to simulate (group size will be half of
%       this)
%    betwgroup_diff: scalar for the between-group difference
% 
% Output:
%    X: the synthetic timeseries, concatenated for the two groups
%    HMM: the HMM fit to the synthetic timeseries (all subjects)
%    features: cell of size 1 x 3 containing the embedded features used 
%       to construct the Fisher kernel, the naive kernel, and the naive
%       normalised kernel
%    Kernel: cell of size 1 x 3 containing the Fisher kernel, the naive 
%       kernel, and the naive normalised kernel, each cell is a samples x 
%       samples kernel matrix
%    err: error in recovering the ground truth group labels in the test set
%
% Christine Ahrends, Aarhus University, 2023

%% Preparation

% set directories
datadir = '/path/to/data'; % this should contain the timeseries (time x regions) for one example subject (synthetic timeseries are then modelled after the dimensions of this example subject)
hmmdir = '/path/to/hmm'; % this should contain a trained HMM (HMM_name) used as a basis for the simulations
scriptdir = '/path/to/code';
hmm_scriptdir = '/path/to/HMM-MAR-master';

addpath(scriptdir)
addpath(genpath(hmm_scriptdir))

% load example subject and HMM
load([datadir '/sub1.mat']) % sub1
load([hmmdir '/' HMM_name '.mat']) % HMM

% normalise example tc and get dimensions
sub1 = normalize(sub1);
n_ts = size(sub1,1);
nregions = size(sub1,2);
T = zeros(1, n_subj);
for i=1:n_subj
    T(1,i) = n_ts;
end
K = HMM.hmm.K; % number of states in example HMM

%% simulate timecourses
rng('shuffle')
        
HMM_group1 = HMM;
T_group1 = T(1:(n_subj/2)); % group 1 will be half the number of subjects simulated
X_group1 = simhmmmar(T_group1, HMM_group1.hmm); % timecourses for group 1

HMM_group2 = HMM_group1; % the basis for group 2 is the same HMM as group 1
clear HMM

% to generate different transition probability matrices, randomly permute
% some of the states' transition probability vectors, how many states
% corresponds to amount of between group difference:

% generate permutations for K-x states
if betwgroup_diff < 1/(K-2)
    noisevec = [1, 3, 2, 4:K];
elseif betwgroup_diff < 2/(K-2)
    noisevec = [1, randperm(3)+1, 5:K];
elseif betwgroup_diff < 3/(K-2)
    noisevec = [1, randperm(4)+1, 6:K];
else 
    noisevec = [1, randperm(5)+1];
end

HMM_group2.hmm.P(1,:) = HMM_group1.hmm.P(1,noisevec); % always shuffle some of the trans probs for state 1
if betwgroup_diff > 4/(K-2) % for larger between-group difference, also shuffle some of the trans probs for state 2
    noisevec2 = [1, 2, randperm(3)+2, 6:K]; 
    HMM_group2.hmm.P(2,:) = HMM_group1.hmm.P(2,noisevec2);
end
if betwgroup_diff > 8/(K-2) % for very large between-group difference, also shuffle some of the trans probs for state 3
    noisevec3 = [1:3, 5, 4, 6:K];
    HMM_group2.hmm.P(3,:) = HMM_group1.hmm.P(3,noisevec3);
end
T_group2 = T(1:(n_subj/2));
X_group2 = simhmmmar(T_group2, HMM_group2.hmm); % timecourses for group 2

X = [X_group1; X_group2]; % concatenate group 1 and group 2

groupsubs = n_subj/2;
Y = ones(n_subj,1);
Y(groupsubs+1:end,1) = Y(groupsubs+1:end,1)+1; % ground truth labels: the first half of subjects are group 1, the second half are group 2

%% fit group-level HMM
% fit new HMM to synthetic timeseries of all subjects
hmm_options = struct();
hmm_options.order = 0;
hmm_options.covtype = 'full'; %('full' for covariance, 'uniquefull' for no covariance)
hmm_options.zeromean = 0; % (0 to model mean, 1 to model only covariance)
hmm_options.standardise = 0; % not necessary here, since data are already standardised
hmm_options.dropstates = 0;
hmm_options.K = K;
hmm_options.useParallel = 0;

% run HMM (group-level)
[HMM.hmm, HMM.Gamma, HMM.Xi, HMM.vpath] = hmmmar(X, T, hmm_options);

%% construct kernels and feature matrices from HMM and classify

Xc = cell(n_subj,1);
for n = 1:n_subj
    Xc{n} = X(sum(T(1:n-1))+1:sum(T(1:n)),:);
end

S = n_subj;

% set kernel options, here **excluding state features**
K_options = struct();
K_options.Pi = true; % state probabilities
K_options.P = true; % transition probabilities
K_options.mu = false; % state means to false
K_options.sigma = false; % state covariances to false
K_options.shape = 'linear';
K_options.normalisation = [];
all_types = {'Fisher', 'naive', 'naive_norm'};

% split into training and test set
ind = 1:S;
split = 0.8; % ratio to split data into training and test set (variable is training set percentage)
train_size = round(S*split);
[train_ind] = randsample(ind, train_size);
test_ind = ind(~ismember(ind, train_ind));

err = zeros(3,1); % initialise empty array to hold errors for all 3 kernels
features = cell(3,1); % initialise cell to hold features
Kernel = cell(3,1); % initialise cell to hold kernels
for kk = 1:3
    K_options.type = all_types{kk};
    [Kernel{kk},features{kk}] = hmm_kernel(Xc, HMM.hmm, K_options); % construct kernels and get feature matrices
    
    train_feat = features{kk}(train_ind,:);
    test_feat = features{kk}(test_ind,:);
    Y_train = Y(train_ind,1); 
    Y_test = Y(test_ind,1);
    
    % fit SVM to predict binary variable in test set from gradient features
    svm = fitcsvm(train_feat, Y_train, 'Standardize', false, 'KernelFunction', 'linear'); % equivalent to using pre-constructed kernels
    [est_labels, ~] = predict(svm, test_feat); % predicted labels
    err(kk,1) = sum(est_labels ~= Y_test) ./numel(Y_test); % error on test set
end

end