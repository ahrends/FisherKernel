function FC_cov = make_staticFCcov(datadir, n_sess)
% FC_cov = make_staticFCcov(datadir, n_sess)
%
% make subject-wise static FC (covariance) matrices for HCP rsFMRI
% This script uses groupICA50 parcellation
% Requires that both the behavioural data and the timecourses in groupICA50
% parcellation can be found in the data directory
%
% Input:
%    datadir: Path to data directory containing behavioural data and
%    groupICA50 timecourses
%    n_sess: The number of sessions per subject to be used (4 to use all
%    sessions)
%
% Output:
%    FC_cov: Covariance matrices for all subjects having at least one
%    behavioural variable (3D tensor of size ROIs x ROIs x subjects)
%
% Christine Ahrends, University of Oxford, 2024

%% Preparation

% load behavioural data to compare indices
all_vars = load([datadir '/vars.txt']);
load([datadir '/headers_grouped_category.mat']) %headers of variables in all_vars
pred_age = all_vars(:,4);
load([datadir '/vars_target_with_IDs.mat'])
int_vars = vars_target_with_IDs;
clear vars_target_with_IDs
target_ind = ismember(all_vars(:,1), int_vars(:,1)); % indices of subjects with at least one behavioural variable (N=1001)

n_subs = 1001; % number of subjects
n_rois = 50; % number of ROIs in parcellation
ts_persess = 1200; % number of timepoints in each session

% load timecourses
tc_std = cell(n_subs,n_sess); % empty cell to hold standardised timecourses
load([datadir '/hcp1003_RESTall_LR_groupICA50.mat'])
data = data(target_ind); % remove subjects missing all behavioural data

%% standardise timecourses and get covariance matrices

% standardise timecourses session-wise
for n = 1:n_subs
    for s = 1:n_sess
        tc_sess = data{n}((s-1)*ts_persess+1:(s-1)*ts_persess+ts_persess,:);
        tc_demean = tc_sess-repmat(mean(tc_sess),size(tc_sess,1),1);
        tc_std{n,s} = tc_demean/std(tc_demean(:));
    end
end

% get covariance matrix
FC_cov = zeros(n_rois, n_rois, n_subs);
for i = 1:n_subs
    cov_tmp = zeros(n_rois, n_rois,n_sess);
    for s = 1:n_sess
        cov_tmp(:,:,s) = cov(tc_std{i,s}); % covariance per session
    end
    FC_cov(:,:,i) = mean(cov_tmp,3); % each subject's covariance matrix is averaged across sessions
end

save([datadir '/FC_cov_groupICA50.mat'], 'FC_cov')

end