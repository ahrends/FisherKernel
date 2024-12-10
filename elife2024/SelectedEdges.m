function results = SelectedEdges(varN, iterN)
%% 
% Selected Edges method for predicting from time-averaged FC matrices, as
% described in Rosenberg et al. 2018 & Shen et al. 2018
% main prediction part adapted from https://www.nitrc.org/projects/bioimagesuite/behavioralprediction.m
%
% Input:
%    varN: variable number (here from 1:35, 1 being age, 2:34 cognitive
%    items)
%    iterN: iteration number, used to load pre-defined folds
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
% Christine Ahrends, Aarhus University, 2023

%% Preparation
% set directories
scriptdir = '/path/to/scripts';
datadir = '/path/to/data';
addpath(genpath([scriptdir '/NetsPredict-master'])) % for consistency with other time-averaged FC analyses, use netspredict for util functions
addpath(scriptdir)
outputdir = '/path/to/results';

% initialise empty struct to hold results
results = struct(); 

% load behavioural data
all_vars = load([datadir '/vars.txt']);
load([datadir '/headers_grouped_category.mat']) %headers of variables in all_vars
pred_age = all_vars(:,4);
load([datadir '/vars_target_with_IDs.mat'])
int_vars = vars_target_with_IDs;
clear vars_target_with_IDs

% find indices of 1,001 subjects for which at least some behavioural data
% are available
target_ind = ismember(all_vars(:,1), int_vars(:,1));
ind_tmp = find(ismember(all_vars(target_ind,:).', int_vars(:,11).', 'rows'));
Y = [pred_age(target_ind),int_vars(:,2:end)];

% find indices and missing subjects for current target variable
Yin = Y(:, varN);
missing_subs = find(isnan(Yin)==true);
index = ~isnan(Yin);
Yin = Yin(index,:); % remove missing subjects from target variable
% concatenate variables to be predicted (here: age and 34 intelligence
% variables)
confounds = all_vars(target_ind,[3,8]);
confounds = confounds(index,:); % remove missing subjects from confounds

% create family structure to use for CV folds (produces variable "twins")
make_HCPfamilystructure; 

cs=twins(index,index); % remove missing subjects from family structure
[allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));  

% load time-averaged FC matrices
load([datadir '/FC_cov_groupICA50.mat']); % load covariance matrices for 1,003 subjects
FC_cov = FC_cov(:,:,target_ind); % remove subjects for which all behavioural variables are missing
all_mats = FC_cov(:,:,index); % remove subjects for which current target variable is missing
clear FC_cov


%% Main: feature selection and regression

thresh = 0.01; % threshold for feature selection

% initialise some variables
no_sub = size(all_mats,3);
no_node = size(all_mats,1);
behav_pred = zeros(no_sub,1);
behav_predD = zeros(no_sub,1);
YD = zeros(no_sub,1);

% load pre-defined CV folds
kfold = 10;
nfolds = 10;
load([maindir '/scripts/Kernel/revisions/kfolds.mat'])
% IMPORTANT!! remove Nan subjects from folds and update the indices!!
for jj = 1:numel(missing_subs)
    snan = missing_subs(jj);
    for ii = 1:nfolds
        folds{iterN}{ii} = [folds{iterN}{ii}(folds{iterN}{ii}<snan), folds{iterN}{ii}(folds{iterN}{ii}>snan)-1]; % folds are loaded above
    end
    missing_subs = missing_subs-1;
end

for i = 1:kfold
    train_mats = all_mats;
    train_mats(:,:,folds{iterN}{i}) = [];
    train_vcts = reshape(train_mats, [], size(train_mats,3));

    % standardise X training set
    train_vctsT = train_vcts';
    N = size(train_mats, 3);
    mx = mean(train_vctsT);  sx = std(train_vctsT);
    train_vctsT = train_vctsT - repmat(mx,N,1);
    train_vctsT(:,sx>0) = train_vctsT(:,sx>0) ./ repmat(sx(sx>0),N,1);
    train_vcts = train_vctsT';
    train_mats = reshape(train_vcts, size(train_mats));
    
    train_behav = Yin;
    train_behav(folds{iterN}{i}) = [];
    
    % Feature selection:
    % correlate all edges with behaviour using robust regression
    edge_no = size(train_vcts,1);
    r_mat = zeros(1, edge_no);
    p_mat = zeros(1, edge_no);
    
    for edge_i = 1:edge_no
        [~, stats] = robustfit(train_vcts(edge_i,:)', train_behav);
        cur_t = stats.t(2);
        r_mat(edge_i) = sign(cur_t)*sqrt(cur_t^2/(no_sub-numel(folds{iterN}{i})-2+cur_t^2)); % adjust degrees of freedom for CV instead of LOO
        p_mat(edge_i) = 2*(1-tcdf(abs(cur_t), no_sub-numel(folds{iterN}{i})-2)); % two-tailed
    end
    
    r_mat = reshape(r_mat, no_node, no_node);
    p_mat = reshape(p_mat, no_node, no_node);
    
    % set threshold and define masks
    pos_mask = zeros(no_node, no_node);
    neg_mask = zeros(no_node, no_node);
    
    pos_edge = find(r_mat > 0 & p_mat < thresh);
    neg_edge = find(r_mat < 0 & p_mat < thresh);
    
    pos_mask(pos_edge) = 1;
    neg_mask(neg_edge) = 1;
    
    % sigmoidal weighting
    % get sum of all edges in train subs (divide by 2 to control for the
    % fact that matrices are symmetric)    
    train_sumpos = zeros(no_sub-numel(folds{iterN}{i}),1);
    train_sumneg = zeros(no_sub-numel(folds{iterN}{i}),1);
    
    for ss = 1:size(train_sumpos)
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    end
    
    % deconfound Y training set
    confounds_tmp = confounds - repmat(mean(confounds),no_sub,1);
    confounds_tmp(folds{iterN}{i},:) = [];
    [~,~,betaY,train_behavD] = ...
                nets_deconfound([],train_behav,confounds_tmp,'gaussian',[],[],[]);
    
    % Main regression:
    % build model on train subs combining both positive and negative
    % features
    b = regress(train_behavD, [train_sumpos, train_sumneg, ones(no_sub-numel(folds{iterN}{i}),1)]);
    
    % run model on test
    test_mat = all_mats(:,:,folds{iterN}{i});
    test_vcts = reshape(test_mat, [], size(test_mat,3));
    
    % standardise X test set
    test_vctsT = test_vcts';
    Ntest = size(test_mat, 3);
    test_vctsT = test_vctsT - repmat(mx,Ntest,1);
    test_vctsT(:,sx>0) = test_vctsT(:,sx>0) ./ repmat(sx(sx>0),Ntest,1);
    test_vcts = test_vctsT';
    test_mat = reshape(test_vcts, size(test_mat));
    
    test_sumpos = sum(sum(test_mat.*pos_mask))/2;
    test_sumneg = sum(sum(test_mat.*neg_mask))/2;
    
    % deconfound Y test set    
    confounds_tmp = confounds - repmat(mean(confounds),no_sub,1);
    confounds_tmp = confounds_tmp(folds{iterN}{i},:);
    test_behav = Yin(folds{iterN}{i});
    [~,~,~,test_behavD] = ...
                nets_deconfound([],test_behav,confounds_tmp,'gaussian',[],betaY,[]);
    
    behav_predD(folds{iterN}{i}) = b(1)*test_sumpos + b(2)*test_sumneg + b(3); % predicted Y in deconfounded space
    
    behav_pred(folds{iterN}{i}) = ...
        nets_confound(behav_predD(folds{iterN}{i}),confounds_tmp,'gaussian',betaY); % predicted Y in original space
    
    % compute errors and assemble results (fold-level):
    % correlation between predicted and true
    results.kcorr_deconf(i) = corr(behav_predD(folds{iterN}{i}), test_behavD); % in deconfounded space
    results.kcorr(i) = corr(behav_pred(folds{iterN}{i}), test_behav); % in original space
    
    % save deconfounded Y
    YD(folds{iterN}{i}) = test_behavD;
    traininds = setdiff(1:no_sub, folds{iterN}{i});
    YD(traininds) = train_behavD;
    
    % normalised maximum absolute errors (in original space)
    errs_deconf = abs(behav_predD(folds{iterN}{i})-test_behavD);
    mae_deconf = max(errs_deconf);
    results.knmae_deconf(i) = mae_deconf/range(YD);
    errs = abs(behav_pred(folds{iterN}{i})-test_behav);
    mae = max(errs);
    results.knmae(i) = mae/range(Yin);
    
    % coefficient of determination
    err_resD = sum((test_behavD - behav_predD(folds{iterN}{i})).^2);
    err_totD = sum((test_behavD - mean(test_behavD)).^2);
    results.kcod_deconf(i) = 1 - (err_resD/err_totD); % in deconfounded space
    err_res = sum((test_behav - behav_pred(folds{iterN}{i})).^2);
    err_tot = sum((test_behav - mean(test_behav)).^2);
    results.kcod(i) = 1 - (err_res/err_tot); % in original space    
    
end

% assemble and save results across folds
results.predictedY = behav_pred;
results.predictedYD = behav_predD;
results.YD = YD;

results.avcorr = mean(results.kcorr);
results.avcorr_deconf = mean(results.kcorr_deconf);

save([outputdir '/SelectedEdges_varN' num2str(varN) '_iterN' num2str(iterN) '.mat'], 'results')

end
