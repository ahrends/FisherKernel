%% Simulations part 1: Sensitivity to different types of features
% 
% Main script to run simulations to test which type of feature differences 
% kernels are able to recover (state means or transition probabilities)
% 
% Uses simulate_statemeans, simulate_transprobs, and
% simulate_transprobs_nostates
%
% Dependencies:
% HMM-MAR toolbox: https://github.com/OHBA-analysis/HMM-MAR
%
% simulate:
% 
% 1. one state's mean different between groups, making sure that the
% between-group difference is smaller than the minimum distance between
% states within the model
%
% 2. difference in transition probabilities 
%
% Christine Ahrends, Aarhus University, 2023

%% Preparation

% set directories
hmmdir = '/path/to/hmm'; % this should contain a trained HMM (HMM_name) used as a basis for the simulations
scriptdir = '/path/to/code';
hmm_scriptdir = '/path/to/HMM-MAR-master';

addpath(scriptdir)
addpath(genpath(hmm_scriptdir))

HMM_name = 'HMM_only_cov0';

% find threshold for simulations where group difference is in state means
% to make sure that difference won't be captured as two different states
% but rather subtle changes in one state:
% load example HMM
load([hmmdir '/' HMM_name '.mat']) % HMM

hmm = HMM.hmm;
K = hmm.K; % number of states in example HMM

% get smallest distance between states in base HMM
for k = 1:K
    for kk = 1:K
        disttmp = abs(hmm.state(k).W.Mu_W-hmm.state(kk).W.Mu_W);
        distMU(k,kk) = mean(disttmp);
    end
end
mindistMU = min(distMU(distMU>0)); % minimum distance between two states mean vectors

rng('shuffle')
jj=1;
nregions = 50;

group_diff = 0;
distnoise = 0;
figure;
while distnoise < mindistMU
    group_diff = group_diff + 0.1;
    noisevec = group_diff(jj) * randn(nregions,1)./4; % add random noise (state inconsistency amount) to state means
    noisestate = hmm.state(1).W.Mu_W + noisevec';
    subplot(1,2,1); imagesc(hmm.state(1).W.Mu_W'); colorbar; title('Original state mean');
    subplot(1,2,2); imagesc(noisestate'); colorbar; title('Noisy state mean');
    distnoise = mean(abs(hmm.state(1).W.Mu_W-noisestate));
end
disp(['Threshold for state mean group difference is ' num2str(group_diff)])
% Note that this will likely be too large (difference between the groups
% would be too obvious so all methods would perform at 100% accuracy)

clear HMM hmm

%% Simulate timecourses and test classification

n_iter = 10; % do this 10 times for each scenario with randomly generated timecourses & random splits
outputdir = '/path/to/results';
if ~isdir(outputdir); mkdir(outputdir); end

% initialise empty arrays to hold results:
err_statemeans = zeros(n_iter, 3);
feat_statemeans = cell(n_iter, 3);
Kernel_statemeans = cell(n_iter, 3);

err_transprobs = zeros(n_iter, 3);
feat_transprobs = cell(n_iter, 3);
Kernel_transprobs = cell(n_iter, 3);

err_transprobs_nostates = zeros(n_iter, 3);
feat_transprobs_nostates = cell(n_iter, 3);
Kernel_transprobs_nostates = cell(n_iter, 3);

n_subj = 200; % simulate 200 subjects (100 in each group)

% simulate group difference in state means and calculate errors in
% recovering group difference across kernels and iterations
betwgroup_diff = 0.2; % relatively subtle between-group difference to see some differences between methods (should be between 0 and threshold defined above)
for i = 1:n_iter
    [X, HMM, features, Kernel, err] = simulate_statemeans(HMM_name, n_subj, betwgroup_diff);
    for kk=1:3
        err_statemeans(i,kk) = err(kk);
        feat_statemeans{i,kk} = features{kk};
        Kernel_statemeans{i,kk} = Kernel{kk};
    end
    save([outputdir '/Sim_statemeans_nsubj' num2str(n_subj) '_groupdiff' num2str(betwgroup_diff) '_iterN' num2str(i) '.mat'], 'X', 'HMM', 'features', 'Kernel', 'err');
    clear X HMM err features Kernel
end

% simulate group difference in transition probabilities and calculate
% errors across kernels and iterations
betwgroup_diff = 1; % relatively subtle difference (should be between 1 and 10)
for i = 1:n_iter
    [X, HMM, features, Kernel, err] = simulate_transprobs(HMM_name, n_subj, betwgroup_diff);
    for kk=1:3
        err_transprobs(i,kk) = err(kk);
        feat_transprobs{i,kk} = features{kk};
        Kernel_transprobs{i,kk} = Kernel{kk};
    end
    save([outputdir '/Sim_transprobs_nsubj' num2str(n_subj) '_groupdiff' num2str(betwgroup_diff) '_iterN' num2str(i) '.mat'], 'X', 'HMM', 'features', 'Kernel', 'err');
    clear X HMM err features Kernel
end

% simulate group difference in transition probabilities, but exclude state
% parameters when constructing the kernels
betwgroup_diff = 1;
for i = 1:n_iter
    [X, HMM, features, Kernel, err] = simulate_transprobs_nostates(HMM_name, n_subj, betwgroup_diff);
    for kk=1:3
        err_transprobs_nostates(i,kk) = err(kk);
        feat_transprobs_nostates{i,kk} = features{kk};
        Kernel_transprobs_nostates{i,kk} = Kernel{kk};
    end
    save([outputdir '/Sim_transprobs_nostates_nsubj' num2str(n_subj) '_groupdiff' num2str(betwgroup_diff) '_iterN' num2str(i) '.mat'], 'X', 'HMM', 'features', 'Kernel', 'err');
    clear X HMM err features Kernel
end

%% Figures

% Error histograms:
figure;
subplot(1,3,1);
[y, b] = hist(err_statemeans, 5);
bar(b,y, 'grouped');
legend({'Fisher kernel', 'Naive kernel', 'Naive norm. kernel'}); 
title('Errors State Mean simulations'); xlim([0, 1]); xlabel('Errors'); axis square;
subplot(1,3,2);
[y, b] = hist(err_transprobs, 5);
bar(b,y, 'grouped');
title('Errors Transition probabilities simulations'); xlim([0, 1]); xlabel('Errors'); axis square;
subplot(1,3,3);
[y, b] = hist(err_transprobs_nostates, 5);
bar(b,y, 'grouped');
title('Errors Transition probabilities (no states) simulations'); xlim([0, 1]); xlabel('Errors'); axis square;

% Example kernels and features for both groups: (here just plotting first
% iteration for each scenario)

% set diagonal of kernels to nan for better visualisation:
Kernel0_statemeans = Kernel_statemeans;
Kernel0_transprobs = Kernel_transprobs;
Kernel0_transprobs_nostates = Kernel_transprobs_nostates;
for i = 1:n_iter
    for kk = 1:3
        for j = 1:n_subj
            Kernel0_statemeans{i,kk}(j,j) = nan;
            Kernel0_transprobs{i,kk}(j,j) = nan;
            Kernel0_transprobs_nostates{i,kk}(j,j) = nan;
        end
    end
end

% State means
figure; 
tiledlayout(3,3,'TileSpacing','loose','Padding','loose');
% Naive kernel:
nexttile([1,2]); plot(feat_statemeans{1,2}(1:100,:)', '.b', 'MarkerSize', 1); hold on; % features group 1
plot(feat_statemeans{1,2}(101,:)', '.r', 'MarkerSize', 1); % features group 2
xticks([1, 7, 43, 343]); xticklabels({''}); xlim([1,size(feat_statemeans{1,1},2)]);  % set ticks for different types of parameters
yticklabels({''}); ylim([-7,7]);
title('Naive features, group1 vs. group2'); 
hold off;
nexttile; imagesc(Kernel0_statemeans{1,2}); % plot kernel
xticklabels({''}); yticklabels({''}); colorbar('TickLabels', ''); axis square;
title(['Naive kernel, error: ' num2str(err_statemeans(1,2))])
% Naive norm. kernel:
nexttile([1,2]); plot(feat_statemeans{1,3}(1:100,:)', '.b', 'MarkerSize', 1); hold on;
plot(feat_statemeans{1,3}(101,:)', '.r', 'MarkerSize', 1); 
xticks([1, 7, 43, 343]); xticklabels({''}); xlim([1,size(feat_statemeans{1,1},2)]); 
yticklabels({''}); ylim([-17,17]);
title('Naive norm. features, group1 vs. group2'); 
hold off;
nexttile; imagesc(Kernel0_statemeans{1,3}); 
xticklabels({''}); yticklabels({''}); colorbar('TickLabels', ''); axis square;
title(['Naive norm. kernel, error: ' num2str(err_statemeans(1,3))])
% Fisher kernel:
nexttile([1,2]); plot(feat_statemeans{1,1}(1:100,:)', '.b', 'MarkerSize', 1); hold on; 
plot(feat_statemeans{1,1}(101,:)', '.r', 'MarkerSize', 1); 
xticks([1, 7, 43, 343]); xticklabels({''}); xlim([1,size(feat_statemeans{1,1},2)]); 
yticklabels({''}); ylim([-230,230]);
title('Fisher scores, group1 vs. group2'); 
hold off;
nexttile; imagesc(Kernel0_statemeans{1,1}); 
xticklabels({''}); yticklabels({''}); colorbar('TickLabels', ''); axis square;
title(['Fisher kernel, error: ' num2str(err_statemeans(1,1))])

% Transition probabilities
figure; 
tiledlayout(3,3,'TileSpacing','loose','Padding','loose');
% Naive kernel:
nexttile([1,2]); plot(feat_transprobs{1,2}(1:100,:)', '.b', 'MarkerSize', 1); hold on;
plot(feat_transprobs{1,2}(101,:)', '.r', 'MarkerSize', 1); 
xticks([1, 7, 43, 343]); xticklabels({''}); xlim([1,size(feat_transprobs{1,1},2)]); 
yticklabels({''}); ylim([-10,10]);
title('Naive features, group 1 vs. group 2')
hold off;
nexttile; imagesc(Kernel0_transprobs{1,2}); 
xticklabels({''}); yticklabels({''}); colorbar('TickLabels', ''); axis square;
title(['Naive kernel, error: ' num2str(err_transprobs(1,2))])
% Naive norm. kernel:
nexttile([1,2]); plot(feat_transprobs{1,3}(1:100,:)', '.b', 'MarkerSize', 1); hold on;
plot(feat_transprobs{1,3}(101,:)', '.r', 'MarkerSize', 1); 
xticks([1, 7, 43, 343]); xticklabels({''}); xlim([1,size(feat_transprobs{1,1},2)]); 
yticklabels({''}); ylim([-17,17]);
title('Naive norm. features, group 1 vs. group 2')
hold off;
nexttile; imagesc(Kernel0_transprobs{1,3}); 
xticklabels({''}); yticklabels({''}); colorbar('TickLabels', ''); axis square;
title(['Naive norm. kernel, error: ' num2str(err_transprobs(1,3))])
% Fisher kernel:
nexttile([1,2]); plot(feat_transprobs{1,1}(1:100,:)', '.b', 'MarkerSize', 1); hold on;
plot(feat_transprobs{1,1}(101,:)', '.r', 'MarkerSize', 1); 
xticks([1, 7, 43, 343]); xticklabels({''}); xlim([1,size(feat_transprobs{1,1},2)]); 
yticklabels({''}); ylim([-266,266]);
title('Fisher scores, group1 vs. group2'); 
holf off;
nexttile; imagesc(Kernel0_transprobs{1,1}); 
xticklabels({''}); yticklabels({''}); colorbar('TickLabels', ''); axis square;
title(['Fisher kernel, error: ' num2str(err_transprobs(1,1))])

% Transition probabilities
figure; 
tiledlayout(3,3,'TileSpacing','loose','Padding','loose');
% Naive kernel:
nexttile([1,2]); plot(feat_transprobs_nostates{1,2}(1:100,:)', '.b', 'MarkerSize', 1); hold on;
plot(feat_transprobs_nostates{1,2}(101,:)', '.r', 'MarkerSize', 1); 
xticks([1, 7, 43, 343]); xticklabels({''}); xlim([1,size(feat_transprobs_nostates{1,1},2)]); 
yticklabels({''}); ylim([-1,1]);
title('Naive features, group 1 vs. group 2')
hold off;
nexttile; imagesc(Kernel0_transprobs_nostates{1,2}); 
xticklabels({''}); yticklabels({''}); colorbar('TickLabels', ''); axis square;
title(['Naive kernel, error: ' num2str(err_transprobs_nostates(1,2))])
% Naive norm. kernel:
nexttile([1,2]); plot(feat_transprobs_nostates{1,3}(1:100,:)', '.b', 'MarkerSize', 1); hold on;
plot(feat_transprobs_nostates{1,3}(101,:)', '.r', 'MarkerSize', 1); 
xticks([1, 7, 43, 343]); xticklabels({''}); xlim([1,size(feat_transprobs_nostates{1,1},2)]); 
yticklabels({''}); ylim([-3,3]);
title('Naive norm. features, group 1 vs. group 2')
hold off;
nexttile; imagesc(Kernel0_transprobs_nostates{1,3}); 
xticklabels({''}); yticklabels({''}); colorbar('TickLabels', ''); clim([20,100]); axis square;
title(['Naive norm. kernel, error: ' num2str(err_transprobs_nostates(1,3))])
% Fisher kernel:
nexttile([1,2]); plot(feat_transprobs_nostates{1,1}(1:100,:)', '.b', 'MarkerSize', 1); hold on;
plot(feat_transprobs_nostates{1,1}(101,:)', '.r', 'MarkerSize', 1); 
xticks([1, 7, 43, 343]); xticklabels({''}); xlim([1,size(feat_transprobs_nostates{1,1},2)]); 
yticklabels({''}); ylim([-60,60]);
title('Fisher scores, group 1 vs. group 2')
hold off;
nexttile; imagesc(Kernel0_transprobs_nostates{1,1}); 
xticklabels({''}); yticklabels({''}); colorbar('TickLabels', ''); axis square;
title(['Fisher kernel, error: ' num2str(err_transprobs_nostates(1,1))])

