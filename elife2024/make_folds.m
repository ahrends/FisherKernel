function folds = make_folds(n_reps, k)

%%
% make randomised repetitions of k folds for CV of 1001 HCP subjects, 
% taking into account family structure and saving folds to be used by
% prediction methods
%
% Input: 
%    n_reps: number of randomised repetitions
%    k: number of folds for each repetition
% Output:
%    folds: folds (cell of size n_reps, each containing k cells with
%    subject indices for the fold)
%
% Christine Ahrends, University of Oxford, 2024

%%
% set directories
scriptdir = '/path/to/code';
hmmscriptdir = '/path/to/HMM-MAR-master';
FK_dir = '/path/to/FisherKernelRepo';
datadir = '/path/to/data';
outputdir = '/path/to/kernels';

addpath(scriptdir)
addpath(hmmscriptdir)
addpath(FK_githubdir)

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

% make folds
CVscheme = [k k];
rng('shuffle')
folds = cell(n_reps,1);
for r = 1:n_reps
    folds{r} = cvfolds_FK(randn(size(Yin,1),1),'gaussian',CVscheme(1),allcs);
end
save([outputdir '/folds.mat'], 'folds');

end