function [predictedY,predictedYD,YD,stats] = predictPhenotype_kernels_kfolds(Yin,Din,options,varargin)
%
% Kernel ridge regression estimation using a distance matrix using (stratified) LOO. 
% Using this means that the HMM was run once, out of the cross-validation loop
% adapted for using either precomputed kernels or building a Gaussian
% kernel from input distance matrices %CA
% adapted for fold-wise model evaluation %CA
%
% INPUT
% Yin       (no. subjects by 1) vector of phenotypic values to predict,
%           which can be continuous or binary. If a multiclass variable
%           is to be predicted, then Yin should be encoded by a
%           (no. subjects by no. classes) matrix, with zeros or ones
%           indicator entries.
% Din       (no. subjects by no. subjects) matrix of either distances between
%           subjects, calculated (for example) by computeDistMatrix or
%           computeDistMatrix_AVFC (in which case, options.kernel should be
%           'Gaussian' or pre-computed kernel, calculated e.g. by hmm_kernel
%           in which case options.kernel should be 'linear';
% options   Struct with the prediction options, with fields:
%   + alpha - for method='KRR', a vector of weights on the L2 penalty on the regression
%           By default: [0.0001 0.001 0.01 0.1 0.4 0.7 1.0 10 100]
%   + sigmafact - for method='KRR', a vector of parameters for the kernel; in particular,
%           this is the factor by which we will multiply a data-driven estimation
%           of the kernel parameter. By default: [1/5 1/3 1/2 1 2 3 5];
%   + K - for method='NN', a vector with the number of nearest neighbours to use
%   + CVscheme - vector of two elements: first is number of folds for model evaluation;
%             second is number of folds for the model selection phase (0 in both for LOO)
%   + CVfolds - prespecified CV folds for the outer loop
%   + biascorrect - whether we correct for bias in the estimation 
%                   (Smith et al. 2019, NeuroImage)
%   + kernel - which kernel to use ('linear' or 'Gaussian')
%   + leaveout_foldN - for fold-wise model evaluation, index of the current
%                       test set fold
%   + verbose -  display progress?
% cs        optional (no. subjects X no. subjects) dependency structure matrix with
%           specifying possible relations between subjects (e.g., family
%           structure), or a (no. subjects X 1) vector defining some
%           grouping, with (1...no.groups) or 0 for no group
% confounds     (no. subjects by  no. of confounds) matrix of features that 
%               potentially influence the phenotypes and we wish to control for 
%               (optional)
%
% OUTPUT
% predictedY    predicted response,in the original (non-decounfounded) space
% predictedYD    predicted response,in the decounfounded space
% YD    response,in the decounfounded space
% stats         structure, with fields
%   + pval - permutation-based p-value, if permutation is run;
%            otherwise, correlation-based p-value
%   + cod - coeficient of determination 
%   + corr - correlation between predicted and observed Y 
%   + baseline_corr - baseline correlation between predicted and observed Y for null model 
%   + sse - sum of squared errors
%   + baseline_sse - baseline sum of squared errors
%   PLUS: All of the above +'_deconf' in the deconfounded space, if counfounds were specified
%   + alpha - selected values for alpha at each CV fold
%   + sigmaf - selected values for sigmafact at each CV fold
%
% Author: Diego Vidaurre, OHBA, University of Oxford
%         Steve Smith, fMRIB University of Oxford

leaveout_foldN = options.leaveout_foldN; % for fold-level model evaluation %CA

if ~isfield(options, 'kernel') % use either pre-computed kernel ('linear') 
    kernel = 'linear';
else
    kernel = options.kernel;
end
if strcmpi(kernel, 'Gaussian') % or distance/divergence matrix for 'gaussian'
    Din(eye(size(Din,1))==1) = 0; 
end
[N,q] = size(Yin);

which_nan = false(N,1); % better to remove NaNs from Yin, Din, confounds, family structure, and folds outside this function %CA
if q == 1
    which_nan = isnan(Yin);
    if any(which_nan)
        Yin = Yin(~which_nan);
        Din = Din(~which_nan,~which_nan);
        warning('NaN found on Yin, will remove...')
    end
    N = size(Yin,1);
end

if nargin < 3 || isempty(options), options = struct(); end

if ~isfield(options,'alpha')
    alpha = [0.0001 0.001 0.01 0.1 0.4 0.7 1.0 10 100];
else
    alpha = options.alpha;
end
if strcmpi(kernel, 'Gaussian')
    if ~isfield(options,'sigmafact')
        sigmafact = [1/5 1/3 1/2 1 2 3 5];
    else
        sigmafact = options.sigmafact;
    end
else
    sigmafact = 1;
end
if ~isfield(options,'K')
    KN = 1:min(50,round(0.5*N));
else
    KN = options.K; 
end

if ~isfield(options,'CVscheme'), CVscheme = [10 10];
else, CVscheme = options.CVscheme; end
if ~isfield(options,'CVfolds'), CVfolds = [];
else, CVfolds = options.CVfolds; end
% if ~isfield(options,'biascorrect'), biascorrect = 0;
% else, biascorrect = options.biascorrect; end
if ~isfield(options,'verbose'), verbose = 0;
else, verbose = options.verbose; end

% check correlation structure
allcs = []; 
if (nargin>3) && ~isempty(varargin{1})
    cs = varargin{1};
    if ~isempty(cs)
        is_cs_matrix = (size(cs,2) == size(cs,1));
        if is_cs_matrix 
            if any(which_nan)
                cs = cs(~which_nan,~which_nan);
            end
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));    
        else
            if any(which_nan)
                cs = cs(~which_nan);
            end
            allcs = find(cs > 0);
        end
    end
else, cs = []; 
end

% get confounds
if (nargin>4) && ~isempty(varargin{2})
    confounds = varargin{2};
    confounds = confounds - repmat(mean(confounds),N,1);
    deconfounding = 1;
    if any(which_nan)
        confounds = confounds(~which_nan,:);
    end
else
    confounds = []; deconfounding = 0;
end

Ymean = zeros(N,q);
YD = zeros(N,q); % deconfounded signal
YmeanD = zeros(N,q); % mean in deconfounded space
predictedY = zeros(N,q);
if deconfounding, predictedYD = zeros(N,q); end

rng('shuffle')
% create the inner CV structure - stratified for family=multinomial
if isempty(CVfolds)
    if CVscheme(1)==1
        folds = {1:N};
    elseif q == 1
        rng('shuffle');
        Yin_copy = Yin; Yin_copy(isnan(Yin)) = realmax;
        folds = cvfolds_FK(Yin_copy,'gaussian',CVscheme(1),allcs); % randomise folds while keeping family structure intact %CA
    else % no stratification
        rng('shuffle');
        folds = cvfolds_FK(randn(size(Yin,1),1),'gaussian',CVscheme(1),allcs);
    end
else
    folds = CVfolds;
end

stats = struct();

%for ifold = 1:length(folds)
ifold = leaveout_foldN; % for fold-level evaluation

if verbose, fprintf('CV iteration %d \n',ifold); end

J = folds{ifold}; % test
%if isempty(J), continue; end
if length(folds)==1
    ji = J;
else
    ji = setdiff(1:N,J); % train
end

D = Din(ji,ji); 
Y = Yin(ji,:);
D2 = Din(J,ji);

% family structure for this fold
Qallcs=[];
if (~isempty(cs))
    if is_cs_matrix
        [Qallcs(:,2),Qallcs(:,1)] = ...
            ind2sub([length(cs(ji,ji)) length(cs(ji,ji))],find(cs(ji,ji)>0));
    else
        Qallcs = find(cs(ji) > 0);
    end
end

for ii = 1:q

    Dii = D; Yii = Y; % in case you use parfor

    ind = find(~isnan(Y(:,ii)));
    Yii = Yii(ind,ii); 
    QDin = Dii(ind,ind); 
    QN = length(ind);
    rng('shuffle')
    Qfolds = cvfolds_FK(Yii,'gaussian',CVscheme(2),Qallcs); % we stratify

    % deconfounding business
    if deconfounding
        Cii = confounds(ji,:); Cii = Cii(ind,:);
        [betaY,interceptY,Yii] = deconfoundPhen(Yii,Cii);
    end

    % centering response
    my = mean(Yii); 
    Yii = Yii - repmat(my,size(Yii,1),1);
    Ymean(J,ii) = my;
    QYin = Yii;

    Dev = Inf(length(alpha),length(sigmafact));

    for isigm = 1:length(sigmafact)

        sigmf = sigmafact(isigm);

        QpredictedY = Inf(QN,length(alpha));
        QYinCOMPARE = QYin;

        % Inner CV loop
        for Qifold = 1:length(Qfolds)

            QJ = Qfolds{Qifold}; Qji=setdiff(1:QN,QJ);
            QD = QDin(Qji,Qji);
            QY = QYin(Qji,:); Qmy = mean(QY); QY = QY-Qmy;
            Nji = length(Qji);
            QD2 = QDin(QJ,Qji);

            if strcmpi(kernel, 'Gaussian')
                sigmabase = auto_sigma(QD);
                sigma = sigmf * sigmabase;
                K = gauss_kernel(QD,sigma);
                K2 = gauss_kernel(QD2,sigma);
            elseif strcmp(kernel, 'linear')
                K = QD;
                K2 = QD2;
            end
            I = eye(Nji);
            ridg_pen_scale = mean(diag(K));

            for ialph = 1:length(alpha)
                alph = alpha(ialph);
                beta = (K + ridg_pen_scale * alph * I) \ QY;
                QpredictedY(QJ,ialph) = K2 * beta + repmat(Qmy,length(QJ),1);
            end
        end

        Dev(:,isigm) = (sum(( QpredictedY - ...
            repmat(QYinCOMPARE,1,length(alpha))).^2) / QN)';

    end

    [~,m] = min(Dev(:)); % Pick the one with the lowest deviance
    [ialph,isigm] = ind2sub(size(Dev),m);
    alph = alpha(ialph);
    Dii = D(ind,ind); D2ii = D2(:,ind);

    if strcmpi(kernel, 'Gaussian')
        sigmf = sigmafact(isigm);
        sigmabase = auto_sigma(D);
        sigma = sigmf * sigmabase;
        K = gauss_kernel(Dii,sigma);
        K2 = gauss_kernel(D2ii,sigma);
    elseif strcmp(kernel, 'linear')
        K = Dii;
        K2 = D2ii;
        sigmf = NaN;
    end
    Nji = length(ind);
    I = eye(Nji);

    ridg_pen_scale = mean(diag(K));
    beta = (K + ridg_pen_scale * alph * I) \ Yii;

    % predict the test fold
    predictedY(J,ii) = K2 * beta + my; % some may be NaN actually

    % predictedYD and YD in deconfounded space; Yin and predictedY are confounded
    predictedYD(J,ii) = predictedY(J,ii);
    YD(J,ii) = Yin(J,ii);
    YmeanD(J,ii) = Ymean(J,ii);
    if deconfounding % in order to later estimate prediction accuracy in deconfounded space
        [~,~,YD(J,ii)] = deconfoundPhen(YD(J,ii),confounds(J,:),betaY,interceptY);
        % original space
        predictedY(J,ii) = confoundPhen(predictedY(J,ii),confounds(J,:),betaY,interceptY);
        Ymean(J,ii) = confoundPhen(YmeanD(J,ii),confounds(J,:),betaY,interceptY);

    end

end

%disp(['Fold ' num2str(ifold) ])
stats.alpha(ifold) = alph;
stats.sigma(ifold) = sigmf;
%    stats.sigma(ifold) = NaN;
%    stats.beta(ifold,:) = beta;

stats.sse = zeros(q,1);
stats.cod = zeros(q,1);
stats.corr = zeros(q,1);
stats.baseline_corr = zeros(q,1);
stats.pval = zeros(q,1);
if deconfounding
    stats.sse_deconf = zeros(q,1);
    stats.cod_deconf = zeros(q,1);
    stats.corr_deconf = zeros(q,1);
    stats.baseline_corr_deconf = zeros(q,1);
    stats.pval_deconf = zeros(q,1);
end

for ii = 1:q
    ind = folds{ifold};
    errs = abs(predictedY(ind,ii)-Yin(ind,ii));
    mae = max(errs);
    stats.nmae(ii) = mae/range(Yin(:,ii));
    stats.sse(ii) = sum((Yin(ind,ii)-predictedY(ind,ii)).^2);
    nullsse = sum((Yin(ind,ii)-Ymean(ind,ii)).^2);
    stats.cod(ii) = 1 - stats.sse(ii) / nullsse;
    stats.corr(ii) = corr(Yin(ind,ii),predictedY(ind,ii));
    stats.baseline_corr(ii) = corr(Yin(ind,ii),Ymean(ind,ii));
    [~,pv] = corrcoef(Yin(ind,ii),predictedY(ind,ii)); % original space
    if corr(Yin(ind,ii),predictedY(ind,ii))<0, stats.pval(ii) = 1;
    else, stats.pval(ii) = pv(1,2);
    end
    if deconfounding
        errs_deconf = abs(predictedYD(ind,ii)-YD(ind,ii));
        mae_deconf = max(errs_deconf);
        stats.nmae_deconf(ii) = mae_deconf/range(YD(:,ii));
        stats.sse_deconf(ii) = sum((YD(ind,ii)-predictedYD(ind,ii)).^2);
        nullsse_deconf = sum((YD(ind,ii)-YmeanD(ind,ii)).^2);
        stats.cod_deconf(ii) = 1 - stats.sse_deconf(ii) / nullsse_deconf;
        stats.corr_deconf(ii) = corr(YD(ind,ii),predictedYD(ind,ii));
        stats.baseline_corr_deconf(ii) = corr(YD(ind,ii),YmeanD(ind,ii));
        [~,pv] = corrcoef(YD(ind,ii),predictedYD(ind,ii)); % deconfounded space
        if corr(YD(ind,ii),predictedYD(ind,ii))<0, stats.pval_deconf(ii) = 1;
        else, stats.pval_deconf(ii) = pv(1,2);
        end
    end
end

end



function K = gauss_kernel(D,sigma)
% Gaussian kernel
D = D.^2; % because distance is sqrt-ed
K = exp(-D/(2*sigma^2));
end


function sigma = auto_sigma (D)
% gets a data-driven estimation of the kernel parameter
D = D(triu(true(size(D,1)),1));
sigma = median(D);
end


function [betaY,my,Y] = deconfoundPhen(Y,confX,betaY,my)
if nargin<3, betaY = []; end
if isempty(betaY)
    my = mean(Y);
    Y = Y - my;
    betaY = (confX' * confX + 0.00001 * eye(size(confX,2))) \ confX' * Y;
else
    Y = Y - my; % added for deconfounding test set (missing in original code) %CA
end
res = Y - confX*betaY;
Y = res;
end


function Y = confoundPhen(Y,conf,betaY,my) 
Y = Y+conf*betaY+my;
end