function results = predict_kernels(datadir, kerneldir, resultsdir, HMM_name, varN, iterN, Fn, Kn)
% results = predict_kernels(datadir, kerneldir, resultsdir, HMM_name, varN, iterN, Fn, Kn)
%
% runs kernel ridge regression to predict behavioural variables (here age
% and intelligence in HCP dataset) using different embeddings/kernels
% from HMM parameters or static FC distance matrices as predictors (see Fn)
% wrapper for predictPhenotype_kernels_kfolds
% (split into small jobs for cluster)
%
% INPUT:
%    datadir: directory for HCP behavioural data and family structure
%    kerneldir: directory where kernels and distance matrices, and 
%       pre-constructed folds can be found
%    resultsdir: (output) directory for prediction results
%    HMM_name: file name for pre-trained HMM
%    varN: variable number (1 for age, 2:35 for intelligence variables)
%    iterN: iteration number (to load pre-defined folds)
%    Fn: select embedding (1 for Fisher, 2 for naive, 3 for naive
%       normalised, 4 for KL divergence (time-varying), 5 for KL divergence
%       (time-averaged), 6 for log-Euclidean)
%    Kn: select shape of kernel (1 for linear, 2 for Gaussian)
%       Note: For KL divergence and log-Euclidean use only Gaussian kernel
%
% OUTPUT (will be written to resultsdir):
%    results: struct containing results
%        kcorr: fold-level correlation between predicted and true Y in original space (1 x k vector)
%        kcorr_deconf: "-" in deconfounded space
%        kcod: fold-level coefficient of determination in original space (1 x k vector)
%        kcod_deconf: "-" in deconfounded space
%        knmae: fold-level normalised maximum absolute error in original space (1 x k vector)
%        knmae_deconf: "-" in deconfounded space
%        predictedY: all test folds predicted Y in original space
%        predictedYD: "-" in deconfounded space
%        YD: deconfounded Y
%        avcorr: correlation between predicted and true Y across folds in original space
%        avcorr_deconf: "-" in deconfounded space
%
% Christine Ahrends, Aarhus University 2022
%
%% Load data
% load X, Y, confounds, family structure, and pre-defined CV folds

% load behavioural data Y and confounds
all_vars = load([datadir '/vars.txt']);
load([datadir '/headers_grouped_category.mat']) %headers of variables in all_vars
pred_age = all_vars(:,4);
load([datadir '/vars_target_with_IDs.mat'])
int_vars = vars_target_with_IDs;
clear vars_target_with_IDs
target_ind = ismember(all_vars(:,1), int_vars(:,1)); % get indices for subjects where at least one behavioural variable is available
confounds = all_vars(target_ind,[3,8]);

% different behavioural variables have missing data for different subjects
% remove NaNs from Y, save index to remove them also from confounds,
% family structure, and pre-defined folds
Y = [pred_age(target_ind),int_vars(:,2:end)];
Yin = Y(:,varN);
missing_subs = find(isnan(Yin)==true); % indices of subjects with NaN for the particular target variable in this run
index = ~isnan(Yin);
Yin = Yin(index,:);

confounds = confounds(index, :); % remove subjects with missing values from confounds

make_HCPfamilystructure; % make family structure (creates variable twins
twins = twins(index, index); % remove subjects with missing values from family structure

% load pre-defined folds for CV (here 100 repetitions of 10 folds)
nfolds = 10;
load([kerneldir '/folds.mat']) % folds
% IMPORTANT!! remove Nan subjects from folds and update the indices!!
for jj = 1:numel(missing_subs)
    snan = missing_subs(jj);
    for ii = 1:nfolds
        folds{iterN}{ii} = [folds{iterN}{ii}(folds{iterN}{ii}<snan), folds{iterN}{ii}(folds{iterN}{ii}>snan)-1]; % folds are loaded above
    end
    missing_subs = missing_subs-1;
end

% embeddings and combinations for kernel ridge regression
all_types = {'Fisher', 'naive', 'naive_norm', 'KL', 'KL_ta', 'Fro'};
all_shapes = {'linear', 'Gaussian'};

type = all_types{Fn};
shape = all_shapes{Kn};

%% main prediction
% make sure to only run this combination if it does not exist already (for
% interrupted runs)
if ~exist([resultsdir '/Results_' type '_' shape '_varN' num2str(varN) 'iterN' num2str(iterN) '.mat'], 'file')
    
    % initialise empty structures to hold results
    results = struct();
    results.predictedY = NaN(size(Yin));
    results.predictedYD = NaN(size(Yin));
    results.YD = NaN(size(Yin));
    
    % load preconstructed kernels (for linear kernels) or
    % distance/divergence matrices (for Gaussian kernels)
    if strcmpi(type, 'KL') % load time-varying KL divergence divergence matrix
        load([kerneldir '/Kernel_' HMM_name '_KLdiv.mat'], 'D'); 
    elseif strcmpi(type, 'KL_ta') % load time-averaged KL divergence matrix
        load([kerneldir '/Kernel_static_KLdiv.mat'])
    elseif strcmpi(type, 'Fro') % load log-Euclidean distance matrix
        load([kerneldir '/Kernel_static_Fro.mat']);
    else % main time-varying kernels
        if strcmpi(shape, 'linear')
            load([kerneldir '/Kernel_' HMM_name '_' type '_' shape '.mat'], 'Kernel'); % load kernel
        elseif strcmpi(shape, 'Gaussian')
            load([kerneldir '/Kernel_' HMM_name '_' type '_' shape '.mat'], 'D'); % load distance matrix
        end
    end
    
    for iii = 1:nfolds 

        % specify options for kernel ridge regression
        krr_params = struct();
        krr_params.deconfounding = 1; % use deconfounding
        krr_params.CVscheme = [10 10]; % number of outer and inner folds for nested CV
        krr_params.CVfolds = folds{iterN}; % use pre-defined folds for outer CV loops (loaded above)
        krr_params.leaveout_foldN = iii; % fold left out for testing in this iteration
        krr_params.alpha = [0.0001 0.001 0.01 0.1 0.3 0.5 0.7 0.9 1.0]; % ridge penalty (vector for grid search)
        krr_params.verbose = 1;
        krr_params.Nperm = 1; 
        krr_params.kernel = shape; % either 'Gaussian' or 'linear';

        rng('shuffle')
        % load predictors: 
        % for linear kernel, load pre-constructed kernel itself, 
        % for Gaussian kernel, load distance/divergence matrix and estimate
        % tau (width of Gaussian kernel) within inner CV
        if strcmpi(shape, 'linear')
            Din = Kernel(index,index);
        elseif strcmpi(shape, 'Gaussian')
            Din = D(index, index);
        end

        [predictedY, predictedYD, YD, stats] = predictPhenotype_kernels_kfolds(Yin, ...
            Din, krr_params, twins, confounds);
    
        % save fold-level results:    
        results.predictedY(folds{iterN}{iii}) = predictedY(folds{iterN}{iii}); % predicted Y in original space
        results.kcorr(iii) = stats.corr; % fold-level correlation coefficient between predicted and true Y in original space
        results.kcod(iii) = stats.cod; % coefficient of determination in original space
        results.knmae(iii) = stats.nmae; % normalised mean absolute error in original space
         
        results.predictedYD(folds{iterN}{iii}) = predictedYD(folds{iterN}{iii}); % predicted Y In deconfounded space
        results.kcorr_deconf(iii) = stats.corr_deconf; % correlation coefficient in deconfounded space
        results.kcod_deconf(iii) = stats.cod_deconf; % coefficient of determination in deconfounded space
        results.knmae_deconf(iii) = stats.nmae_deconf; % normalised maximum absolute error in deconfounded space
        results.YD(folds{iterN}{iii}) = YD(folds{iterN}{iii}); % deconfounded Y

    end
   
    results.avcorr = corr(results.predictedY, Yin); % correlation coefficient in original space across folds
    results.avcorr_deconf = corr(results.predictedYD, results.YD); % correlation coefficient in deconfounded space across folds
        
    % write results to outputdir
    if ~isdir(resultsdir); mkdir(resultsdir); end
    save([resultsdir '/Results_' type '_' shape '_varN' num2str(varN) '_iterN' num2str(iterN) '.mat'], 'results');
    
end
end
