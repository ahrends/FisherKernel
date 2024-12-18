function folds = make_folds(datadir, kerneldir, n_reps, n_folds)
% folds = make_folds(datadir, kerneldir, n_reps, n_folds)
%
% make randomised repetitions of k folds for CV of 1001 HCP subjects, 
% taking into account family structure and saving folds to be used by
% prediction methods
%
% Input: 
%    datadir: directory for HCP behavioural variables and family structure
%    kerneldir: (output) directory where folds will be saved
%    n_reps: number of randomised repetitions
%    n_folds: number of folds for each repetition
% Output:
%    folds: folds (cell of size n_reps, each containing k cells with
%    subject/sample indices for the fold)
%
% Christine Ahrends, University of Oxford, 2024

%% Load data (Y and family structure)

% load behavioural data to get subject indices
all_vars = load([datadir '/vars.txt']);
load([datadir '/headers_grouped_category.mat']) %headers of variables in all_vars
pred_age = all_vars(:,4);
load([datadir '/vars_target_with_IDs.mat'])
int_vars = vars_target_with_IDs;
clear vars_target_with_IDs
target_ind = ismember(all_vars(:,1), int_vars(:,1));

Yin = pred_age(target_ind);
index = ~isnan(Yin);
Yin = Yin(index,:); % there should not be NaNs for age (just double-checking)

% get family structure
make_HCPfamilystructure; % create variable twins to hold family structure
allcs = []; 
cs = twins(index, index);
allcs = find(cs > 0);

%% Make folds

CVscheme = [n_folds n_folds];
rng('shuffle')
folds = cell(n_reps,1);
for r = 1:n_reps
    folds{r} = cvfolds_FK(randn(size(Yin,1),1),'gaussian',CVscheme(1),allcs); % folds contains n_reps cells, each containing n_folds with subject indices
end

if ~isdir(kerneldir); mkdir(kerneldir); end

save([kerneldir '/folds.mat'], 'folds');

end