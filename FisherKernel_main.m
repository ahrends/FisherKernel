%% Fisher Kernel: Main script

%% Preparation
% set directories and general variables, load data

projectdir = '/user/FisherKernel_project';
scriptdir = [projectdir '/scripts/FisherKernel']; % directory for this folder
hmm_scriptdir = [projectdir '/scripts/HMM-MAR-master'];
datadir = [projectdir '/data/HCP_1200']; % directory where HCP S1200 timecourses and behavioural/demographic variables can be found
outputdir = [projectdir '/results'];

cd(projectdir)
addpath(genpath(scriptdir))
addpath(genpath(hmm_scriptdir));

% load Y: variables to be predicted and confounds
% should be a subjects x variables matrix
all_vars = load([datadir '/vars.txt']);
load([datadir '/headers_grouped_category.mat']) % headers of variables in all_vars
load([datadir '/vars_target_with_IDs.mat'])
int_vars = vars_target_with_IDs;
clear vars_target_with_IDs
target_ind = ismember(all_vars(:,1), int_vars(:,1)); % find indices of subjects for which we have int_vars
confounds = all_vars(target_ind,[3,8]);
% create family structure to use for CV folds (produces variable "twins")
make_HCPfamilystructure;
% concatenate variables to be predicted (here: age and 34 intelligence
% variables)
Y = [pred_age(target_ind),int_vars(:,2:end)];

% load X: timecourses (here called 'data') that the HMM should be run on
load([datadir '/tcs/hcp1003_RESTall_LR_groupICA50.mat']);
% assuming here that timecourses are a subjects x 1 cell, each containing a
% timepoints x ROIs matrix

data_X = data(target_ind);
clear data all_vars headers_grouped_category int_vars

S = size(data_X,1); % number of subjects (for which at least one of the variables we want to predict are available)
N = size(data_X{1},2); % number of ROIs
T = cell(size(data_X,1),1); % cell containing no. of timepoints for each subject
for s = 1:S
    T{s} = size(data_X{s},1); % should be 4800 when using all four scanning sessions, 1200 when using only on
end

%% 1. Run group-level HMM

% Main text version: Gaussian HMM where states have mean and
% covariance run on all 4 scanning sessions per participant

K = 6; % number of HMM-states
hmm_options = struct();
hmm_options.order = 0; % Gaussian
hmm_options.covtype = 'full'; %('full' for covariance, 'uniquefull' for no covariance)
hmm_options.zeromean = 0; % (0 to model mean, 1 to model only covariance)
hmm_options.standardise = 1; 
hmm_options.dropstates = 0;
hmm_options.K = K;
% hmm_options.useParallel = 0; 
[HMM.hmm, HMM.Gamma, HMM.Xi, HMM.vpath, ~, ~, HMM.fehist] = hmmmar(data_X, T, hmm_options);
if ~isdir(outputdir); mkdir(outputdir); end
save([outputdir '/HMM_main.mat'], 'HMM')

% (Other versions for Supplementary Information)
% SI version 1: only first scanning session of each participant:
load([datadir '/tcs/hcp1003_REST1_LR_groupICA50.mat']);
data_X1 = data(target_ind);
T1 = cell(size(data_X1,1),1); % cell containing no. of timepoints for each subject
for s = 1:S
    T1{s} = size(data_X1{s},1); % should be 4800 when using all four scanning sessions, 1200 when using only on
end
clear HMM
[HMM.hmm, HMM.Gamma, HMM.Xi, HMM.vpath, ~, ~, HMM.fehist] = hmmmar(data_X1, T1, hmm_options);
save([outputdir '/HMM_rest1.mat'], 'HMM')

% SI version 2: Gaussian HMM where states have only covariance (mean set to
% 0):
hmm_options.zeromean = 1; % (0 to model mean, 1 to model only covariance)
clear HMM
[HMM.hmm, HMM.Gamma, HMM.Xi, HMM.vpath, ~, ~, HMM.fehist] = hmmmar(data_X, T, hmm_options);
save([outputdir '/HMM_cov.mat'], 'HMM')

%% 2. Construct kernels (and/or embedded features or distance/divergence matrices)

% (load HMM)
HMM_version = 'HMM_main'; % change to run on SI versions of HMM
load([outputdir '/' HMM_version '.mat'])

% construct kernels and feature matrices from HMM
% decide which parameters to use:
% Pi: state probabilities
% P: transition probabilities
% mu: state means
% sigma: state covariances
% at the moment: use either a) Pi & P, b) Pi, P, and sigma, or c) Pi, P, mu, and sigma
% we here used all available parameters, i.e. Pi, P, mu, and sigma for main
% text version and Pi, P, and sigma for HMM where states have only
% covariance
K_options = struct();
K_options.Pi = true;
K_options.P = true;
if HMM.hmm.train.zeromean==0
    K_options.mu = true; % use state means only if they were estimated
else
    K_options.mu = false; % otherwise use only state covariances
end
K_options.sigma = true;

Fnames = {'naive', 'naive_norm', 'Fisher'};
Knames = {'linear', 'gaussian'};

% This will construct the kernels and the features (compute the gradient
% for Fisher kernel) and run dual estimation (Fisher-compatible version)
for Fn=1:3
    for Kn=1:2
        K_options.type = Fnames{Fn}; % one of 'naive', 'naive_norm', or 'Fisher'
        % 'naive' will also give the vectorised parameters, 'naive_norm' will also
        % give the normalised vectorised parameters, 'Fisher' will also give the
        % gradient features
        K_options.kernel = Knames{Kn};

        if ~isdir(outputdir); mkdir(outputdir); end
        clear Kernel features Dist
        if Kn==2
            [Kernel, features, Dist] = hmm_kernel(data_X, HMM.hmm, K_options);
            save([outputdir '/Kernel_' Fnames{Fn} '_' Knames{Kn} '.mat'], 'Kernel', 'features', 'Dist');
        else
            [Kernel, features] = hmm_kernel(data_X, HMM.hmm, K_options);
            save([outputdir '/Kernel_' Fnames{Fn} '_' Knames{Kn} '.mat'], 'Kernel', 'features');
        end
    end
end

% compare also to KL divergence model
clear Dist
Dist = computeDistMatrix(data_X, T, HMM.hmm);
save([outputdir '/Kernel_KLdiv.mat'], 'Dist');

% (SI: compare also to static FC KL divergence model)
clear Dist
Dist = computeDistMatrix_AVFC(data_X,T);
save([outputdir '/Kernel_KLdiv_staticFC.mat'], 'Dist')

%% 3. Run KRR for prediction
% set up variables & options for KRR
N_variables = 35;
N_iter = 1; % in the paper, this is set to 100;

krr_params = struct();
krr_params.deconfounding = 1;
krr_params.CVscheme = [10 10];
krr_params.alpha = [0.0001 0.001 0.01 0.1 0.3 0.5 0.7 0.9 1.0];
krr_params.verbose = 1;
krr_params.Nperm = 1; % 100 (for permutation-based significance testing)

for Fn=1:3
    for Kn=1:2
        for varN = 1:N_variables
            for iterN = 1:N_iter
                if ~exist([outputdir '/Predictions_' HMM_version '_' Fnames{Fn} '_' Knames{Kn} '_varN' num2str(varN) 'iterN' num2str(iterN) '.mat'], 'file')
                    % load kernel (for linear kernel) or
                    % distance/divergence matrix (for Gaussian kernel)
                    if strcmpi(Fnames{Fn}, 'KL')
                        load([outputdir '/Kernel_KLdiv.mat'], 'Dist'); % load KL divergence matrix
                    else
                        if strcmpi(Knames{Kn}, 'linear')
                            load([outputdir '/Kernel_' Fnames{Fn} '_' Knames{Kn} '.mat'], 'Kernel');
                        elseif strcmpi(kernel, 'gaussian')
                            load([outputdir '/Kernel_' Fnames{Fn} '_' Knames{Kn} '.mat'], 'Dist');
                        end
                    end
                    krr_params.kernel = Knames{Kn}; %'gaussian','linear';
                    results = struct();
                    Yin = Y(:,varN);
                    index = ~isnan(Yin);
                    Yin = Yin(index,:);
                    rng('shuffle')
                    % choose input: use Kernel for linear kernel and distance matrix for
                    % Gaussian kernel (to estimate kernel width within KRR CV)
                    if strcmpi(Knames{Kn}, 'linear')
                        Din = Kernel(index,index);
                    elseif strcmpi(Knames{Kn}, 'gaussian')
                        Din = Dist(index, index);
                    end
                    % disp(['Now running iteration ' num2str(i) ' out of ' num2str(niter) ...
                    %    ' for variable ' num2str(j) ' out of ' num2str(nvars) ' for ' ...
                    %       Knames{k} ' ' Fnames{f} ' kernel']);
                    [results.predictedY, results.predictedYD, results.YD, results.stats] = predictPhenotype_kernels(Yin, ...
                        Din, krr_params, twins(index,index), confounds(index,:)); % twins is the family structure used for CV
    
                    if ~isdir(outputdir); mkdir(outputdir); end
                    save([outputdir '/Predictions_' HMM_version '_' Fnames{Fn} '_' Knames{Kn} '_varN' num2str(varN) 'iterN' num2str(iterN) '.mat'], 'results');
                end
            end
        end
    end
end

% (SI: run also for static FC KL divergence)
load([outputdir '/Kernel_KLdiv_staticFC.mat'], 'Dist');
krr_params.kernel = 'gaussian';
Din = Dist(index, index);
for varN = 1:N_variables
    for iterN = 1:N_iter
        results = struct();
        Yin = Y(:,varN);
        index = ~isnan(Yin);
        Yin = Yin(index,:);
        rng('shuffle')
        % disp(['Now running iteration ' num2str(i) ' out of ' num2str(niter) ...
        %    ' for variable ' num2str(j) ' out of ' num2str(nvars) ' for static FC KL']);
        [results.predictedY, results.predictedYD, results.YD, results.stats] = predictPhenotype_kernels(Yin, ...
            Din, krr_params, twins(index,index), confounds(index,:));
        if ~isdir(outputdir); mkdir(outputdir); end
        save([outputdir '/Predictions_' HMM_version '_staticFCKL_varN' num2str(varN) 'iterN' num2str(iterN) '.mat'], 'results');
    end
end

%% 4. load and evaluate predictions
% load all predictions, evaluate correlation between predicted and actual,
% NRMSE, and NMAXAE, and assemble everything in a table

Fnames = {'naive', 'naive_norm', 'Fisher', 'staticFC'};
kernel_resultsT = table();
i= 0;
for Fn = 1:5
    for Kn = 1:2
        for varN = 1:N_variables
            for iterN = 1:N_iter
                if ~((Fn>3) && (Kn==1))
                    if Fn~=5
                        load([outputdir '/Predictions_' HMM_version '_' Fnames{Fn} '_' Knames{Kn} '_varN' num2str(varN) 'iterN' num2str(iterN) '.mat']);
                    else
                        load([outputdir '/Predictions_' HMM_version '_staticFCKL_varN' num2str(varN) 'iterN' num2str(iterN) '.mat']);
                    end
                    i = i+1;
                    % settings for this prediction
                    kernel_resultsT.features{i} = Fnames{Fn};
                    kernel_resultsT.kernel{i} = Knames{Kn};
                    kernel_resultsT.varN(i) = varN;
                    kernel_resultsT.iterN(i) = iterN;
                    kernel_resultsT.predictedY{i} = results.predictedY;
                    % for sanity checks and optimisation, check betas (if
                    % saved from KRR) and hyperparameters 
                    kernel_resultsT.beta{i} = results.beta; % beta are the regression weights
                    kernel_resultsT.lambda{i} = results.stats.alpha; % regularisation parameter is called alpha in predictPhenotype, but lambda in the manuscript
                    kernel_resultsT.tau{i} = results.stats.sigma; % width of Gaussian kernel (this parameter is irrelevant for linear kernels), 
                    % note that this is called sigma in predictPhenotype, but tau in the manuscript to avoid confusion with the HMM state covariances
                    kernel_resultsT.corr(i) = results.stats.corr; % Pearson's r betw. model-predicted and actual
                    kernel_resultsT.err{i} = kernel_resultsT.predictedY{i} - Y(~isnan(Y(:,varN)),varN);
                    kernel_resultsT.RMSE(i) = sqrt(mean(kernel_resultsT.err{i})^2);
                    kernel_resultsT.NRMSE(i) = kernel_resultsT.RMSE(i)/range(Y(:,varN));
                    kernel_resultsT.MAXAE(i) = max(abs(kernel_resultsT.err{i}(:)));
                    kernel_resultsT.NMAXAE(i) = kernel_resultsT.MAXAE(i)/range(Y(:,varN));
                    clear results
                end
            end
        end
    end
end

save([outputdir '/kernel_resultsT.mat'], 'kernel_resultsT');

% robustness: standard deviation over 100 iterations of
% 10-fold CV for each variable and each kernel
robustnessT = table();
all_measures = {'corr', 'NRMSE'};
j=1;
for Fn = 1:4
    for Kn = 1:2
        for v = 1:varN
            for m = 1:2
                features = Fnames{Fn};
                kernel = Knames{Kn};
                measure = all_measures{m};
                if ~((Fn==4) && (Kn==1))
                    robustnessT.features{j} = features;
                    robustnessT.kernel{j} = kernel;
                    robustnessT.measure{j} = measure;
                    robustnessT.varN(j) = v;
                    robustnessT.std(j) = std(eval(['kernel_resultsT.' measure ...
                        '((strcmpi(kernel_resultsT.kernel, kernel)) & (strcmpi(kernel_resultsT.features, features)) & (kernel_resultsT.varN == v))']));
                    j = j+1;
                end
            end
        end
    end
end

save([outputdir '/Kernel_robustnessT.mat'], 'robustnessT')

%% 5. Test for differences between kernels
% permutation tests for differences in accuracy and robustness

% set up which comparisons to do
% logical indices for results table
CN = strcmp(kernel_resultsT.features, 'naive');
CNN = strcmp(kernel_resultsT.features, 'naive_norm');
CFK = strcmp(kernel_resultsT.features, 'Fisher');
CKL = strcmp(kernel_resultsT.features, 'KL');
Clin = strcmp(kernel_resultsT.kernel, 'linear');
Cgaus = strcmp(kernel_resultsT.kernel, 'gaussian');

% set up comparisons: each row is a test, mat1 and mat2 are the logical
% indices to be compared
% compare Fisher kernel with naive kernel and naive normalised kernel
% (linear & Gaussian versions collapsed)
n_comp = 12;
compsT = table();
compsT.mat1{1} = CN;
compsT.mat2{1} = CNN;
compsT.mat1{2} = CN;
compsT.mat2{2} = CFK;
compsT.mat1{3} = CNN;
compsT.mat2{3} = CFK;
% compare linear with Gaussian versions of kernels
compsT.mat1{4} = CN & Clin;
compsT.mat2{4} = CN & Cgaus;
compsT.mat1{5} = CNN & Clin;
compsT.mat2{5} = CNN & Cgaus;
compsT.mat1{6} = CFK & Clin;
compsT.mat2{6} = CFK & Cgaus;
% compare each kernel with KL divergence
compsT.mat1{7} = CN & Clin;
compsT.mat1{8} = CN & Cgaus;
compsT.mat1{9} = CNN & Clin;
compsT.mat1{10} = CNN & Cgaus;
compsT.mat1{11} = CFK & Clin;
compsT.mat1{12} = CFK & Cgaus;
for iii = 7:12
    compsT.mat2{iii} = CKL;
end

design = [ones([1,35]) repmat(2,[1,35])]';
n_perm = 5000;
Pvals_acc = zeros(2,n_comp); % row 1 will be correlation, row 2 will be results for NRMSE, columns are the comparisons set up above

% test for significant differences in accuracy (correlation coefficient and
% NRMSE)
for ii = 1:n_comp
    %tic
    clear mat1 mat2 data
    for n = 1:varN
        mat1(n,:) = kernel_resultsT.corr(compsT.mat1{ii} & kernel_resultsT.varN==n)';
        mat2(n,:) = kernel_resultsT.corr(compsT.mat2{ii} & kernel_resultsT.varN==n)';
    end
    data = [mat1; mat2];   
    Pvals_acc(1,ii) = mean(permtest_aux(data,design,n_perm));
    clear mat1 mat2 data
    for n = 1:varN
        mat1(n,:) = kernel_resultsT.NRMSE(compsT.mat1{ii} & kernel_resultsT.varN==n)';
        mat2(n,:) = kernel_resultsT.NRMSE(compsT.mat2{ii} & kernel_resultsT.varN==n)';
    end
    data = [mat1; mat2];  
    Pvals_acc(2,ii) = mean(permtest_aux(data,design,n_perm));
    %toc
end

% Benjamini-Hochberg corrected p-values:
for i = 1:2
    [Pvals_acc_sort(i,:), ind(i,:)] = sort(Pvals_acc(i,:));
    for j = 1:size(Pvals_acc_sort,2)-1
        Pvals_acc_BHtmp(i,j) = min(Pvals_acc_sort(i,j)*n_comp,Pvals_acc_sort(i,j+1));
    end
    Pvals_acc_BHtmp(i,12) = Pvals_acc_sort(i,j)*n_comp;
    for j = 1:size(Pvals_acc_sort,2)
        Pvals_acc_BH(i,ind(i,j)) = Pvals_acc_BHtmp(i,j);
    end
end


save([outputdir '/Permtests_acc.mat', 'Pvals_acc', 'Pvals_acc_BH');    
    
% test for significant differences in robustness of accuracy (standard 
% deviation of correlation coefficient and NRMSE)

% comparisons are the same as above

design = [ones([1,35]) repmat(2,[1,35])]';
n_perm = 5000;
Pvals_rob = zeros(1,n_comp); 

for ii = 1:n_comp
    %tic
    clear mat1 mat2 data
    for n = 1:2
        mat1(:,n) = robustnessT.std(compsT.mat1{ii} & strcmp(robustnessT.measure, all_measures{n}));
        mat2(:,n) = robustnessT.std(compsT.mat2{ii} & strcmp(robustnessT.measure, all_measures{n}));
    end
    if ii < 4
        data = [[mat1(1:35,:), mat1(36:end,:)]; [mat2(1:35,:), mat2(36:end,:)]];
    else
        data = [mat1; mat2]; 
    end
    Pvals_rob(1,ii) = mean(permtest_aux(data,design,n_perm));
end

% Benjamini-Hochberg corrected p-values:
[Pvals_rob_sort(1,:), indrob(1,:)] = sort(Pvals_rob(1,:));
for j = 1:size(Pvals_rob_sort,2)-1
    Pvals_rob_BHtmp(1,j) = min(Pvals_rob_sort(1,j)*n_comp,Pvals_rob_sort(1,j+1));
end
Pvals_rob_BHtmp(1,12) = Pvals_rob_sort(1,j)*n_comp;
for j = 1:size(Pvals_rob_sort,2)
    Pvals_rob_BH(1,indrob(1,j)) = Pvals_rob_BHtmp(1,j);
end
        
save([outputdir '/Permtests_rob.mat'], 'Pvals_rob', 'Pvals_rob_BH');    
    
%% 6. Calculate risk of extreme errors
% for each kernel, calculate percent of all runs where the *normalised* 
% maximum absolute error exceeded 10, 100, and 1,000 respectively

maxerr_sum10 = zeros(4,2); % rows are "features" (i.e. naive, naive normalised, Fisher, and KL divergence) and columns are versions of the kernels (linear and Gaussian)
maxerr_sum100 = zeros(4,2);
maxerr_sum1000 = zeros(4,2);
for Fn = 1:4
    for Kn = 1:2
        features = Fnames{Fn};
        kernel = Knames{Kn};
        if ~(Fn==4 && Kn==1)
            maxerr_sum10(Fn,Kn) = sum(kernel_resultsT.NMAXAE(strcmp(kernel_resultsT.features, features) & strcmp(kernel_resultsT.kernel, kernel))>10);
            maxerr_sum100(Fn,Kn) = sum(kernel_resultsT.NMAXAE(strcmp(kernel_resultsT.features, features) & strcmp(kernel_resultsT.kernel, kernel))>100);
            maxerr_sum1000(Fn,Kn) = sum(kernel_resultsT.NMAXAE(strcmp(kernel_resultsT.features, features) & strcmp(kernel_resultsT.kernel, kernel))>1000);
        end
    end
end
maxerr10_risk = maxerr_sum10./3500; % 3500 runs (35 variables * 100 iterations of CV) in total for each kernel
maxerr100_risk = maxerr_sum100./3500;
maxerr1000_risk = maxerr_sum1000./3500;

save([outputdir 'maxerr_risk.mat', 'maxerr10_risk', 'maxerr100_risk', 'maxerr1000_risk');    
