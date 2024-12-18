function D = build_logEuclidean(datadir, kerneldir)
% D = build_logEuclidean(datadir, kerneldir)
% 
% build log Euclidean distance matrix
% This is the Frobenius norm of the logarithm map of the time-averaged 
% covariance matrices (Jayasumana et al., arXiv 2013)
% The distance matrix will be used in Gaussian kernel for KRR in main
% prediction function
%
% Input:
%    datadir: directory for HCP rsFMRI static FC covariances matrices
%    kerneldir: (output) directory where kernels and features will be saved
% 
% Output:
%    D: distance matrix
%
% Christine Ahrends, University of Oxford, 2024

%% Preparation

% load static covariance matrices (output of make_staticFCcov)
load([datadir '/FC_cov_groupICA50.mat']) % FC_cov

%% compute distance matrix
n_subs = size(FC_cov, 3);
D = zeros(n_subs);

for i = 1:n_subs
    for j = 1:n_subs
        D(i,j) = norm((log(squeeze(FC_cov(:,:,i)))-log(squeeze(FC_cov(:,:,j)))), 'fro');
    end
end

% to check that distance matrix looks reasonable
% figure; imagesc(D_fro); axis square; colorbar
save([kerneldir '/Kernel_static_Fro.mat'], 'D')

end