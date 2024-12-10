function D = build_logEuclidean(FC_cov)
%% 
% build log Euclidean distance matrix
% This is the Frobenius norm of the logarithm map of the time-averaged 
% covariance matrices (Jayasumana et al., arXiv 2013)
% The distance matrix will be used in Gaussian kernel for KRR in main
% prediction function
%
% Input:
%    FC_cov: covariance matrices (3D matrix of size ROIs x ROIs x subjects)
% 
% Output:
%    D: distance matrix
%
% Christine Ahrends, University of Oxford, 2024

%% Preparation

% set directories
datadir = '/path/to/data';
outputdir = '/path/to/kernels';

% load behavioural data (to get correct subject indices)
all_vars = load([datadir '/vars.txt']);
load([datadir '/headers_grouped_category.mat']) % headers of variables in all_vars
pred_age = all_vars(:,4);
load([datadir '/vars_target_with_IDs.mat'])
int_vars = vars_target_with_IDs;
clear vars_target_with_IDs
target_ind = ismember(all_vars(:,1), int_vars(:,1)); % indices of 1,001 subjects with at least one behavioural variable

FC_cov = FC_cov(:,:,target_ind); % remove subjects missing behavioural data

%% compute distance matrix
n_subs = 1001;
D = zeros(n_subs);

for i = 1:n_subs
    for j = 1:n_subs
        D(i,j) = norm((log(squeeze(FC_cov(:,:,i)))-log(squeeze(FC_cov(:,:,j)))), 'fro');
    end
end

% to check that distance matrix looks reasonable
% figure; imagesc(D_fro); axis square; colorbar
save([outputdir '/Kernel_static_Fro.mat'], 'D')

end