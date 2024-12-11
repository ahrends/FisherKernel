function results = predict_static(varN, iterN, EN, Riem)

%%
% Prediction using (non-kernelised) static FC features with Ridge 
% regression/Elastic Net. Output is fold-wise.
% Wrapper for nets_predict5_kfold
% 
% Dependencies:
% NetsPredict - https://github.com/vidaurre/NetsPredict
% covariancetoolbox - https://github.com/alexandrebarachant/covariancetoolbox
% 
% Input:
%    varN: variable number (1 for age, 2:35 for intelligence variables)
%    iterN: iteration number (to load pre-defined folds)
%    EN: use Elastic Net? (1 for Elastic Net, 0 for ridge regression)
%    Riem: in Riemannian space (use tangent space projection)? (1 for
%    Riemannian, 0 for Euclidean)
% 
% Output:
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

%% Preparation

% set directories
scriptdir = '/path/to/code';
datadir = '/path/to/data'; % this should contain the behavioural data and pre-computed static covariance matrices
npdir = '/path/to/NetsPredict-master';
covdir = '/path/to/covariancetoolbox-master';
kerneldir = '/path/to/kernels'; % should contain the pre-defined folds
outputdir = '/path/to/results';

addpath(scriptdir)
addpath(genpath(npdir))
addpath(genpath(covdir))

if ~isdir(outputdir); mkdir(outputdir); end

% set names for results files
if EN==0
    type = 'statFC_RR';
elseif EN==1
    type = 'statFC_EN';
end

if Riem==0
    Riem_char = '';
elseif Riem==1
    Riem_char = 'Riem';
end

% just setting these for consistency with time-varying results
only_cov = 0;
shape = 'linear';

% only run this combination if it does not exist yet (for interrupted runs)
if ~exist([outputdir '/Results_only_cov' num2str(only_cov) '_' type Riem_char '_' shape '_varN' num2str(varN) 'iterN' num2str(iterN) '.mat'], 'file')

    nfolds = 10;

    %% load data
    % load behavioural data
    all_vars = load([datadir '/vars.txt']);
    load([datadir '/headers_grouped_category.mat']) %headers of variables in all_vars
    pred_age = all_vars(:,4);
    load([datadir '/vars_target_with_IDs.mat'])
    int_vars = vars_target_with_IDs;
    clear vars_target_with_IDs
    target_ind = ismember(all_vars(:,1), int_vars(:,1));
    ind_tmp = find(ismember(all_vars(target_ind,:).', int_vars(:,11).', 'rows'));
    Y = [pred_age(target_ind),int_vars(:,2:end)]; % concatenate variables to be predicted (here: age and 34 intelligence variables)
    Yin = Y(:, varN);
    missing_subs = find(isnan(Yin)==true);
    index = ~isnan(Yin);
    Yin = Yin(index,:);

    confounds = all_vars(target_ind,[3,8]);
    confounds = confounds(index,:);

    %% prepare variables for run

    % create family structure to use for CV folds (produces variable "twins")
    make_HCPfamilystructure; 
    twins = twins(index,index);

    load([kerneldir '/folds.mat']) % folds
    % IMPORTANT!! remove Nan subjects from folds and update the indices!!
    for jj = 1:numel(missing_subs)
        snan = missing_subs(jj);
        for ii = 1:nfolds
            folds{iterN}{ii} = [folds{iterN}{ii}(folds{iterN}{ii}<snan), folds{iterN}{ii}(folds{iterN}{ii}>snan)-1]; % folds are loaded above
        end
        missing_subs = missing_subs-1;
    end

    %% main prediction

    % to run in Euclidean space, load the *unwrapped* static FC matrices, to
    % run in Riemannian space, load the (3D) tensors
    if Riem==0
        load([datadir '/FC_cov_groupICA50.mat']); % FC_cov
        FC_cov = FC_cov(:,:,target_ind);
        ind_all = find(index);
        for i = 1:numel(ind_all)
            Xin_tmp = triu(squeeze(FC_cov(:,:,ind_all(i))));
            Xin_tmptmp = unwrap(Xin_tmp(Xin_tmp~=0));
            Xin(i,:) = Xin_tmptmp; % subject x unwrapped ROIs
        end
    elseif Riem==1
        load([datadir '/FC_cov_groupICA50.mat']);
        FC_cov = FC_cov(:,:,target_ind);
        Xin_tmp = FC_cov(:,:,index);
        Xin = permute(Xin_tmp, [3, 1,2]); % subject x ROI x ROI
    end

    % initialise struct to hold results
    results = struct();
    results.predictedY = NaN(size(Yin));
    results.predictedYD = NaN(size(Yin));
    results.YD = NaN(size(Yin));

    for iii = 1:nfolds

        % set parameters for regression
        parameters = [];
        parameters.CVfolds = folds{iterN}; % use pre-defined folds (loaded above)
        parameters.leaveout_foldN = iii; % for fold-wise model evaluation, specify current test fold
        if EN==0
            parameters.Method = 'ridge';
        elseif EN==1
            parameters.Method = 'glmnet';
        end
        parameters.riemann = Riem;
        parameters.CVscheme = [10 10];
        parameters.deconfounding = [0 1];
        parameters.verbose = 1;

        rng('shuffle')

        [stats, predictedY, ~, predictedYD, ~, ~, YD] = nets_predict5_kfold(Yin, Xin, 'gaussian', parameters, twins, [], confounds);

        % model outcomes in original space:
        results.predictedY(folds{iterN}{iii}) = predictedY(folds{iterN}{iii}); % predicted Y for this fold
        results.kcorr(iii) = stats.corr; % fold-level correlation coefficient between model-predicted and true values
        results.kcod(iii) = stats.cod; % fold-level coefficient of determination
        results.knmae(iii) = stats.nmae; % fold-level normalised maximum absolute error 
        % model outcomes in deconfounded space:
        results.predictedYD(folds{iterN}{iii}) = predictedYD(folds{iterN}{iii}); % predicted Y for this fold
        results.kcorr_deconf(iii) = stats.corr_deconf; % fold-level correlation coefficient between model-predicted and true values
        results.kcod_deconf(iii) = stats.cod_deconf; % fold-level coefficient of determination
        results.knmae_deconf(iii) = stats.nmae_deconf; % fold-level normalised maximum absolute error
        results.YD(folds{iterN}{iii}) = YD(folds{iterN}{iii}); % predicted Y for this fold

    end

    results.avcorr = corr(results.predictedY, Yin); % correlation coefficient in original space across folds
    results.avcorr_deconf = corr(results.predictedYD, results.YD); % correlation coefficient in deconfounded space across folds

    save([outputdir '/Results_only_cov' num2str(only_cov) '_' type Riem_char '_' shape '_varN' num2str(varN) 'iterN' num2str(iterN) '.mat'], 'results');

end
end
