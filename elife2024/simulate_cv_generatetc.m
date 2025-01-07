function [X, Y_true] = simulate_cv_generatetc(datadir, hmmdir, outputdir, HMM_name, n_train, n_test, betwgroup_diff, Y_noise)
% [X, Y_true] = simulate_cv_generatetc(HMM_name, n_train, n_test, betwgroup_diff, Y_noise)
%
% simulate timecourses where the target variable is related to one HMM
% state's means, but with a group difference (heterogeneous training and
% test set) related to another state's means. This will use states #1 and 
% #2 from the example HMM, so any HMM with k>=2 will work.
% 
% Dependencies:
% HMM-MAR toolbox: https://github.com/OHBA-analysis/HMM-MAR
%
% Input: 
%    HMM_name: root file name of the HMM to be loaded (this will be the
%       basis for generating synthetic timecourses)
%    n_train: number of subjects to simulate for the training set
%    n_test: number of subjects to simulate for the test set
%    betwgroup_diff: scalar for the difference between the training and 
%       test set. Note that if this is chosen to be too large, the HMM 
%       will assign a new state to the second group and drop the other 
%       state rather than capturing the difference as a change in the same 
%       state
%    Y_noise: noise on the target variable (if 0, target variable is
%       perfectly correlated with one state's means; larger values make 
%       target variable harder to predict)
%
% Output:
%   X: simulated timecourses for all subjects (first the training, then the
%       test subjects)
%   Y_true: true values for target variable
%
% Christine Ahrends, University of Oxford, 2024

%% Load example data

% load example subject and HMM
load([datadir '/sub1.mat']) % sub1
load([hmmdir '/' HMM_name '.mat']) % HMM

% normalise example tc and get dimensions
sub1 = normalize(sub1);
n_ts = size(sub1,1);
nregions = size(sub1,2);

n_subj = n_train + n_test;
T = zeros(1, n_subj);
for i=1:n_subj
    T(1,i) = n_ts;
end

%% simulate timecourses
rng('shuffle')

% introduce difference between training and test set into mean of state #1
HMM_group1 = HMM;
clear HMM

HMM_group2 = HMM_group1;
noisevec = betwgroup_diff * randn(nregions,1)./4; % add random noise to state means
HMM_group2.hmm.state(1).W.Mu_W = HMM_group1.hmm.state(1).W.Mu_W + noisevec'; % group 2's state 1 is group 1's state 1 + noise

% introduce target difference into mean of state #2
Y_true = randn(n_subj,1); % generate random target variable
HMM_indiv = cell(n_subj,1); % need to create individual HMMs for all subjects
for s = 1:n_subj
    if s <= n_train % if subject is in training set, use HMM_group1 as basis
        HMM_indiv{s} = HMM_group1;
    else % if subject is in test set, use HMM_group2
        HMM_indiv{s} = HMM_group2;
    end
    noisevec2 = Y_noise * randn(nregions,1)./8; % generate noise for target variable
    HMM_indiv{s}.hmm.state(2).W.Mu_W = HMM_indiv{s}.hmm.state(2).W.Mu_W + (Y_true(s)/8) + noisevec2'; % add noise and target variable to state means (scaling each by /8 should be appropriate)
end

% simulate data for individual subjects (train and test) from their HMMs
X = zeros(n_subj*n_ts, nregions);
for s = 1:n_subj
    X_tmp = simhmmmar(T(s), HMM_indiv{s}.hmm);
    X((s-1)*n_ts+1:(s-1)*n_ts+n_ts,:) = X_tmp;
end

if ~isfolder(outputdir); mkdir(outputdir); end
save([outputdir '/X_ntrain' num2str(n_train) '_ntest' num2str(n_test) ...
    '_betwgroupdiff' num2str(betwgroup_diff) '_Ynoise' num2str(Y_noise) '.mat'], "X", "T", "n_ts", "n_subj");
save([outputdir '/Y_ntrain' num2str(n_train) '_ntest' num2str(n_test) ...
    '_betwgroupdiff' num2str(betwgroup_diff) '_Ynoise' num2str(Y_noise) '.mat'], "Y_true");

end