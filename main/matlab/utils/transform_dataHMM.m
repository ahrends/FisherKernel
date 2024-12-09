function [data, T] = transform_dataHMM(data, T, options)
%
% Helper function to transform data according to embeddings/transformations
% used in the HMM
% Christine Ahrends, Aarhus University 2022

if iscell(T)
    if size(T,1)==1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
end
N = length(T);

if xor(iscell(data),iscell(T)), error('X and T must be cells, either both or none of them.'); end

% data can be a cell or a matrix
if isstruct(data) && isfield(data,'C') && ...
        isfield(options,'nessmodel') && options.nessmodel
    warning('data.C and options.nessmodel are not compatible')
    data = data.X;
end
if iscell(T)
    T = cell2mat(T);
end
checkdatacell;

[options,data] = checkoptions(options,data,T,0);
checkData(data,T,options);

% Standardise data and control for ackward trials
    valid_dims = computeValidDimensions(data,options);
    data = standardisedata(data,T,options.standardise,valid_dims); 
    % Filtering
    if ~isempty(options.filter)
       data = filterdata(data,T,options.Fs,options.filter);
    end
    % Detrend data
    if options.detrend
       data = detrenddata(data,T); 
    end
    % Leakage correction
    if options.leakagecorr ~= 0 
        data = leakcorr(data,T,options.leakagecorr);
    end
    % Hilbert envelope
    if options.onpower
       data = rawsignal2power(data,T); 
    end
    % Leading Phase Eigenvectors 
    if options.leida
        data = leadingPhEigenvector(data,T);
    end
    % pre-embedded PCA transform
    if length(options.pca_spatial) > 1 || (options.pca_spatial > 0 && options.pca_spatial ~= 1)
        if isfield(options,'As')
            data.X = bsxfun(@minus,data.X,mean(data.X));   
            data.X = data.X * options.As; 
        else
            [options.As,data.X] = highdim_pca(data.X,T,options.pca_spatial);
            options.pca_spatial = size(options.As,2);
        end
    end    
    % Embedding
    if length(options.embeddedlags) > 1  
        [data,T] = embeddata(data,T,options.embeddedlags);
        elmsg = '(embedded)';
    else
        elmsg = ''; 
    end
    % PCA transform
    if length(options.pca) > 1 || (options.pca > 0 && options.pca ~= 1) || ...
            isfield(options,'A')
        if isfield(options,'A')
            data.X = bsxfun(@minus,data.X,mean(data.X));   
            data.X = data.X * options.A; 
        else
            [options.A,data.X,e] = highdim_pca(data.X,T,options.pca,0,0,0,options.varimax);
            options.pca = size(options.A,2);
            if options.verbose
                if options.varimax
                    fprintf('Working in PCA/Varimax %s space, with %d components. \n',elmsg,options.pca)
                    fprintf('(explained variance = %1f)  \n',e(options.pca))
                else
                    fprintf('Working in PCA %s space, with %d components. \n',elmsg,options.pca)
                    fprintf('(explained variance = %1f)  \n',e(options.pca))
                end
            end
        end
        % Standardise principal components and control for ackward trials
        data = standardisedata(data,T,options.standardise_pc);
        options.ndim = size(options.A,2);
        options.S = ones(options.ndim);
        options.Sind = formindexes(options.orders,options.S);
        if ~options.zeromean, options.Sind = [true(1,size(options.Sind,2)); options.Sind]; end
    else
        options.ndim = size(data.X,2);
    end
    % Downsampling
    if options.downsample > 0 
       [data,T] = downsampledata(data,T,options.downsample,options.Fs); 
    end
    if options.pcamar > 0 && ~isfield(options,'B')
        % PCA on the predictors of the MAR regression, per lag: X_t = \sum_i X_t-i * B_i * W_i + e
        options.B = pcamar_decomp(data,T,options);
    end
    if options.pcapred > 0 && ~isfield(options,'V')
        % PCA on the predictors of the MAR regression, together: 
        % Y = X * V * W + e, where X contains all the lagged predictors
        % So, unlike B, V draws from the temporal dimension and not only spatial
        options.V = pcapred_decomp(data,T,options);
    end   
    
end