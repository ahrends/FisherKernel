function [agg_table] = collect_results(resultsdir, options)
% [agg_table] = collect_results(resultsdir, options)
%
% assemble results from all methods & runs into tables for stats testing &
% figures
% Tables are written out as csv
%
% Input:
% resultsdir: (output) directory for results to be written
% options: struct containing fields:
%    + main: to assemble main results
%    + featuresets: to assemble feature sets results
%    + CV: to assemble HMM training scheme results
%
% Output (will be written to resultsdir):
%    agg_table: struct containing:
%       resultsT: table containing main results from all 14 methods
%       featsets_resultsT: table containing results from runs comparing
%           different feature sets (real data)
%       CV_resultsT: table containing results from runs comparing HMM training
%           schemes (training HMM on all subjects vs. only training set)
%
% Christine Ahrends, University of Oxford, 2024

%% Main results:
% load data from all runs into a table

if isfield(options, 'main') && options.main
    
    all_types = {'Fisher', 'naive', 'naive_norm', 'KL', 'KL_ta', 'Fro', 'statFC_RR', 'statFC_RRRiem', 'statFC_EN', 'statFC_ENRiem', 'SelectedEdges'};
    all_shapes = {'linear', 'Gaussian'};
    
    nmethods = 14;
    nvars = 35;
    niter = 100;
    nfolds = 10;
    
    nrows=nmethods*nvars*niter*nfolds;
    
    % initialise empty table to fill row-wise
    resultsT = table(cell(nrows,1), cell(nrows,1), zeros(nrows,1), ...
        zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), ...
        zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), ...
        'VariableNames', {'type', 'shape', 'varN', 'iterN', 'foldN', ...
        'kcorr', 'kcorr_deconf', 'kcod', 'kcod_deconf', 'knmae', 'knmae_deconf'});
    
    i = 0;
    for Fn = 1:3 % main kernels
        type = all_types{Fn};
        for Kn = 1:2
            shape = all_shapes{Kn};
            for varN = 1:nvars
                for iterN = 1:niter
                    load([resultsdir '/Results_' type '_' shape '_varN' num2str(varN) '_iterN' num2str(iterN) '.mat']); % results
                    for k = 1:nfolds
                        i = i+1;
                        resultsT.type{i} = type;
                        resultsT.shape{i} = shape;
                        resultsT.varN(i) = varN;
                        resultsT.iterN(i) = iterN;
                        resultsT.foldN(i) = k;
                        resultsT.kcorr(i) = results.kcorr(k);
                        resultsT.kcorr_deconf(i) = results.kcorr_deconf(k);
                        resultsT.kcod(i) = results.kcod(k);
                        resultsT.kcod_deconf(i) = results.kcod_deconf(k);
                        resultsT.knmae(i) = results.knmae(k);
                        resultsT.knmae_deconf(i) = results.knmae_deconf(k);
                    end
                end
            end
        end
    end
    for Fn = 4:11
        if Fn <=6
            shape = 'Gaussian';
        else 
            shape = 'linear';
        end
        type = all_types{Fn};
        for varN = 1:nvars
            for iterN = 1:niter
                load([resultsdir '/Results_' type '_' shape '_varN' num2str(varN) '_iterN' num2str(iterN) '.mat']); % results
                for k = 1:nfolds
                    i = i+1;
                    resultsT.type{i} = type;
                    resultsT.shape{i} = shape;
                    resultsT.varN(i) = varN;
                    resultsT.iterN(i) = iterN;
                    resultsT.foldN(i) = k;
                    resultsT.kcorr(i) = results.kcorr(k);
                    resultsT.kcorr_deconf(i) = results.kcorr_deconf(k);
                    resultsT.kcod(i) = results.kcod(k);
                    resultsT.kcod_deconf(i) = results.kcod_deconf(k);
                    resultsT.knmae(i) = results.knmae(k);
                    resultsT.knmae_deconf(i) = results.knmae_deconf(k);
                end
            end
        end
    end
    
    writetable(resultsT, [resultsdir '/MAINresultsT.csv'])
end

%% Results feature sets:

if isfield(options, 'featuresets') && options.featuresets

    all_types = {'Fisher', 'naive', 'naive_norm'};
    all_featsets = {'full', 'noPiP', 'nostates', 'PCAstates'};
    
    nvars = 35;
    niter = 100;
    nfolds = 10;
    nmethods = numel(all_types);
    nfeatsets = numel(all_featsets);
    
    nrows=nmethods*nfeatsets*nvars*niter*nfolds;
    
    featsets_resultsT = table(cell(nrows,1), cell(nrows,1), cell(nrows,1), zeros(nrows,1), ...
        zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), ...
        zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), ...
        'VariableNames', {'type', 'shape', 'featureset', 'varN', 'iterN', 'foldN', ...
        'kcorr', 'kcorr_deconf', 'kcod', 'kcod_deconf', 'knmae', 'knmae_deconf'});
    
    i = 0;
    shape = 'linear';
    for Fn = 1:nmethods
        type = all_types{Fn};
        for FSn = 1:nfeatsets
            featureset = all_featsets{FSn};
            for varN = 1:nvars
                for iterN = 1:niter
                    if FSn == 1
                        % for full runs, load main results
                        load([resultsdir '/Results_' type '_' shape '_varN' num2str(varN) '_iterN' num2str(iterN) '.mat']); % results
                    else % reduced feature sets
                        load([resultsdir '/Results_' featureset '_' type '_' shape '_varN' num2str(varN) '_iterN' num2str(iterN) '.mat']); % results
                    end
                    for k = 1:nfolds
                        i = i+1;
                        featsets_resultsT.type{i} = type;
                        featsets_resultsT.shape{i} = shape;
                        featsets_resultsT.featureset{i} = featureset;
                        featsets_resultsT.varN(i) = varN;
                        featsets_resultsT.iterN(i) = iterN;
                        featsets_resultsT.foldN(i) = k;
                        featsets_resultsT.kcorr(i) = results.kcorr(k);
                        featsets_resultsT.kcorr_deconf(i) = results.kcorr_deconf(k);
                        featsets_resultsT.kcod(i) = results.kcod(k);
                        featsets_resultsT.kcod_deconf(i) = results.kcod_deconf(k);
                        featsets_resultsT.knmae(i) = results.knmae(k);
                        featsets_resultsT.knmae_deconf(i) = results.knmae_deconf(k);
                    end
                end
            end
        end
    end
    
    writetable(featsets_resultsT, [resultsdir '/FEATSETSresultsT.csv'])
end

%% Results training scheme:
% training HMM together vs. separate:
if isfield(options, 'CV') && options.CV

    all_types = {'Fisher', 'naive', 'naive_norm'};
    all_schemes = {'tog', 'sep'};
    all_shapes = {'linear', 'Gaussian'};
    
    nvars = 35;
    nfolds = 10;
    nmethods = numel(all_types);
    nschemes = numel(all_schemes);
    nkernels = numel(all_shapes);
    
    nrows=nmethods*nschemes*nkernels*nvars*nfolds; % doing this only for one iteration (rather than 100 as above) because of HMM training cost
    
    CV_resultsT = table(cell(nrows,1), cell(nrows,1), cell(nrows,1), zeros(nrows,1), ...
        zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), ...
        zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), zeros(nrows,1), ...
        'VariableNames', {'type', 'shape', 'training', 'varN', 'iterN', 'foldN', ...
        'kcorr', 'kcorr_deconf', 'kcod', 'kcod_deconf', 'knmae', 'knmae_deconf'});
    
    i = 0;
    iterN = 1;
    for Fn = 1:nmethods
        type = all_types{Fn};
        for Kn = 1:nkernels
            shape = all_shapes{Kn};
            for CVn = 1:nschemes
                CV = all_schemes{CVn};
                for varN = 1:nvars
                    load([resultsdir '/Results_' CV '_' type '_' shape '_varN' num2str(varN) '_iterN' num2str(iterN) '.mat']); % results              
                    for k = 1:nfolds
                        i = i+1;
                        CV_resultsT.type{i} = type;
                        CV_resultsT.shape{i} = shape;
                        CV_resultsT.training{i} = CV;
                        CV_resultsT.varN(i) = varN;
                        CV_resultsT.iterN(i) = iterN;
                        CV_resultsT.foldN(i) = k;
                        CV_resultsT.kcorr(i) = results.kcorr(k);
                        CV_resultsT.kcorr_deconf(i) = results.kcorr_deconf(k);
                        CV_resultsT.kcod(i) = results.kcod(k);
                        CV_resultsT.kcod_deconf(i) = results.kcod_deconf(k);
                        CV_resultsT.knmae(i) = results.knmae(k);
                        CV_resultsT.knmae_deconf(i) = results.knmae_deconf(k);
                    end
                end
            end
        end
    end
    
    writetable(CV_resultsT, [resultsdir '/CVresultsT.csv'])
end

agg_table = struct();
if isfield(options, 'main') && options.main
    agg_table.main = resultsT;
end
if isfield(options, 'featuresets') && options.featuresets
    agg_table.featuresets = featsets_resultsT;
end
if isfield(options, 'CV') && options.CV
    agg_table.CV = CV_resultsT;
end

end