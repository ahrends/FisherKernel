%% Simulations part 2: Sensitivity to heterogeneous training and test sets
%
% Main script to run simulations to test how predictions from different
% kernels are affected by fitting the HMM either only to training subjects
% or to all subjects, when train and test subjects are taken from different
% distributions. 
%
% Uses simulate_cv_generatetc, fit_HMM_simcv, and predict_simcv
%
% Dependencies:
% HMM-MAR toolbox: https://github.com/OHBA-analysis/HMM-MAR
%
% simulate:
% timecourses for a training set and a test set with varying degrees of 
% between-set heterogeneity, contained in a single state's mean vector
% target variable for all subjects, contained in another state's mean
% vector (with varying degrees of noise)
% fit HMM to only training subjects or all subjects
% predict simulated target variable from Fisher kernel and naive kernels
% constructed from HMMs trained only on training subjects or all subjects
%
% Christine Ahrends, University of Oxford, 2024

%% Preparation

% set directories
datadir = '/path/to/data'; % to store the simulated timecourses and target variables
hmmdir = '/path/to/hmm'; % this should contain one example pre-trained HMM and will be used to store the fitted HMMs
outputdir = '/path/to/results'; % to store the results
scriptdir = '/path/to/code';
hmm_scriptdir = '/path/to/HMM-MAR-master';

addpath(scriptdir)
addpath(genpath(hmm_scriptdir))

%% Main simulations
% generate timecourses and target variables, 
% fit HMMs (only training subjects vs. all subjects),
% construct kernels and predict target variable

example_HMM = 'HMM_only_cov0';
output_HMM = 'HMM_simcv';
k = 6;
n_train = 50;
n_test = 50;
betwgroup_diff = 0.1:0.2:0.9;
Y_noise = 0.5:0.2:1.9;

for i = 1:numel(betwgroup_diff)
    for j = 1:numel(Y_noise)
        simulate_cv_generatetc(example_HMM, n_train, n_test, betwgroup_diff(i), Y_noise(j));
        for cv = 1:2
            fit_HMM_simcv(output_HMM, n_train, n_test, betwgroup_diff(i), Y_noise(j), k, cv);
            predict_simcv(output_HMM, n_train, n_test, betwgroup_diff(i), Y_noise(j), cv);
        end
    end
end

%% assemble results

types = {'Fisher', 'naive', 'naive norm.'};
cv = {'sep', 'tog'};
results_simcv = table();

n = 1;
for ii = 1:numel(betwgroup_diff)
    for jj = 1:numel(Y_noise)
        for tt = 1:2
            load([outputdir '/Results_' output_HMM '_' cv{tt} '_ntrain' ...
                num2str(n_train) '_ntest' num2str(n_test) '_betwgroupdiff' ...
                num2str(betwgroup_diff(ii)) '_Ynoise' num2str(Y_noise(jj)) '.mat']);            
            for kk = 1:3
                results_simcv.kernel_name{n} = types{kk};
                results_simcv.kernel{n} = Kernel{kk};
                results_simcv.feat{n} = features{kk};
                results_simcv.corr_test(n) = corr_test(kk);
                results_simcv.betwgroup_diff(n) = betwgroup_diff(ii);
                results_simcv.Y_noise(n) = Y_noise(jj);
                results_simcv.train_size(n) = n_train;
                results_simcv.test_size(n) = n_test;
                results_simcv.training_type{n} = cv{tt};
                n = n+1;
            end
        end
    end
end

%% Figures

% Figure 6B: Simulations: Accuracies by training scheme
% left panel swarm chart
jitter = randn(240,1);

figure;
for i=1:6:235
    jitter(i+3) = jitter(i+3)+10;
    plot(jitter([i,i+3]), table2array(results_simcv([i,i+3],4)), 'r-o'); hold on;
end
for ii=2:6:236
    jitter(ii+3) = jitter(ii+3)+10;
    plot(jitter([ii,ii+3]), table2array(results_simcv([ii, ii+3],4)), 'g-o'); hold on;
end
for iii=3:6:237
    jitter(iii+3) = jitter(iii+3)+10;
    plot(jitter([iii, iii+3]), table2array(results_simcv([iii, iii+3],4)), 'b-o'); hold on;
end
hold off;

% right panel 2 x 2 scatter plots
condSep = strcmp(results_simcv.training_type, 'sep');
condTog = strcmp(results_simcv.training_type, 'tog');
for nn = 1:size(results_simcv,1)
    if strcmpi(results_simcv.kernel_name(nn), 'Fisher')
        results_simcv.kernel_number(nn) = 1;
    elseif strcmpi(results_simcv.kernel_name(nn), 'naive')
        results_simcv.kernel_number(nn) = 2;
    elseif strcmpi(results_simcv.kernel_name(nn), 'naive norm.')
        results_simcv.kernel_number(nn) = 3;
    end
end

figure; 
subplot(2,2,1); 
swarmchart(results_simcv(condSep,:), "betwgroup_diff", "corr_test", "filled", ...
    "ColorVariable", "kernel_number"); title("Training separate"); axis square;
subplot(2,2,2);
swarmchart(results_simcv(condTog,:), "betwgroup_diff", "corr_test", "filled", ...
    "ColorVariable", "kernel_number"); title("Training together"); axis square;
subplot(2,2,3);
swarmchart(results_simcv(condSep,:), "Y_noise", "corr_test", "filled", ...
    "ColorVariable", "kernel_number"); title("Training separate"); axis square;
subplot(2,2,4);
swarmchart(results_simcv(condTog,:), "Y_noise", "corr_test", "filled", ...
    "ColorVariable", "kernel_number"); title("Training together"); axis square;

% Figure 6C: Simulations: High group-difference kernels
figure; 
for kk=1:3
    subplot(2,3,kk); imagesc(results_simcv.kernel{234+kk}); axis square; colorbar; title([results_simcv.kernel_name(234+kk) ' Training ' results_simcv.training_type(234+kk)]); 
    subplot(2,3,kk+3); imagesc(results_simcv.kernel{234+kk+3}); axis square; colorbar; title([results_simcv.kernel_name(234+kk+3) ' Training ' results_simcv.training_type(234+kk+3)])
end