### Fisher Kernel: elife 2024 Statistics and figures

##### Preparation #####
# install and load relevant packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(correctR, ggplot2, Cairo, ggpubr, lawstat, forcats, ggstatsplot)

resultsdir <- "/path/to/results" # this should point to the directory where matlab results tables were exported to
# should contain the files MAINresultsT.csv, FEATSETSresultsT.csv, and CVresultsT.csv
setwd(resultsdir)

# load results tables
results_main <- read.csv("MAINresultsT.csv", header=TRUE)
results_featsets <- read.csv("FEATSETSresultsT.csv", header=TRUE)
results_CV <- read.csv("CVresultsT.csv", header=TRUE)

##### Data and table formatting #####
###### Main results #####
# adjust variable types, classes, and orders: for main results
results_main$features <- factor(
  results_main$features, 
  levels=c('Fisher', 'naive', 'naive_norm', 'KL', 'KL_ta', 'Fro', 'statFC_RR', 
           'statFC_RRRiem', 'statFC_EN', 'statFC_ENRiem', 'SelectedEdges'))
results_main$kernel <- factor(results_main$kernel, levels=c('linear','Gaussian'))
results_main$varN2 <- as.factor(results_main$varN)
varN3 <- c("Age", "MMSE Score", "PicSeq_Unadj", "PicSeq_AgeAdj", "CardSort_Unadj",
           "CardSort_AgeAdj", "Flanker_Unadj", "Flanker_AgeAdj", "PMAT24_A_CR",
           "PMAT24_A_SI", "PMAT_A_RTCR", "ReadEng_Unadj", "ReadEng_AgeAdj",
           "PicVocab_Unadj", "PicVocab_AgeAdj", "ProcSpeed_Unadj", "ProcSpeed_AgeAdj",
           "VSPLOT_TC", "VSPLOT_CRTE", "VSPLOT_OFF", "SCPT_TP", "SCPT_TN", "SCPT_FP",
           "SCPT_FN", "SCPT_TRPRT", "SCPT_SEN", "SCPT_SPEC", "SCPT_LRNR",
           "IWRD_TOT", "IWRD_RTC", "ListSort_Unadj", "ListSort_AgeAdj", 
           "Language_Task_Acc", "Relational_Task_Acc", "WM_Task_Acc")
results_main$varN3 <- varN3[results_main$varN] # actual variable names

results_main$model <- results_main$features:results_main$kernel
results_main$model <- droplevels(results_main$model) # drop unused combinations
results_main$model <- factor(results_main$model, levels = c(
  "Fisher:linear", "naive:linear", "naive_norm:linear",
  "Fisher:Gaussian", "naive:Gaussian", "naive_norm:Gaussian", "KL:Gaussian",
  "KL_ta:Gaussian", "Fro:Gaussian", "statFC_RR:linear", "statFC_RRRiem:linear",
  "statFC_EN:linear", "statFC_ENRiem:linear", "SelectedEdges:linear"
)) # reorder for plotting

results_main$modelvar <- results_main$model:results_main$varN2 # for grouping by method and target variable

###### Summary table (across folds) ######
# every 10 rows in the table are individual folds for the same model run
kfold_table <- results_main[seq(1, nrow(results_main), 10),]
kfold_table <- subset(kfold_table, select = -c(foldN, kcorr, kcorr_deconf, kcod, kcod_deconf, knmae, knmae_deconf))
kfold_table$kcorr <- colMeans(matrix(results_main$kcorr, nrow=10), na.rm=TRUE)
kfold_table$kcorr_deconf <- colMeans(matrix(results_main$kcorr_deconf, nrow=10), na.rm=TRUE)
kfold_table$kcod <- colMeans(matrix(results_main$kcod, nrow=10), na.rm=TRUE)
kfold_table$kcod_deconf <- colMeans(matrix(results_main$kcod_deconf, nrow=10), na.rm=TRUE)
kfold_table$knmae <- colMeans(matrix(results_main$knmae, nrow=10), na.rm=TRUE) # **average** NMAE
kfold_table$knmae_deconf <- colMeans(matrix(results_main$knmae_deconf, nrow=10), na.rm=TRUE)
# gets model performance for each run, averaged over folds
# making sure to remove missing values because Elastic Net failed to converge in many runs

# for normalised maximum absolute error, use maximum across folds instead of mean:
n <- nrow(results_main)
g <- rep(1:n, each = 10, length = n)
kfold_table$knmae2 <- tapply(results_main$knmae, g, max)
kfold_table$knmae_deconf2 <- tapply(results_main$knmae_deconf, g, max)


###### Feature sets results #####
results_featsets$features <- factor(
  results_featsets$features, 
  levels=c('Fisher', 'naive', 'naive_norm'))
results_featsets$featureset <- factor(
  results_featsets$featureset,
  levels=c('full', 'nostates', 'PCAstates', 'noPiP')
)
results_featsets$fullname <- results_featsets$features:results_featsets$featureset # for grouping by method and feature subset
results_featsets$varN2 <- as.factor(results_featsets$varN)
results_featsets$varN3 <- varN3[results_featsets$varN] # actual variable names

results_featsets$modelvar <- results_featsets$fullname:results_featsets$varN2 # grouping by method and target variable

###### Summary table feature sets #####
# as above, summarise performance across folds:
kfold_feats <- results_featsets[seq(1, nrow(results_featsets), 10),]
kfold_feats <- subset(kfold_feats, select = -c(foldN, kcorr, kcorr_deconf, kcod, kcod_deconf, knmae, knmae_deconf))
kfold_feats$kcorr <- colMeans(matrix(results_featsets$kcorr, nrow=10), na.rm=TRUE)
kfold_feats$kcorr_deconf <- colMeans(matrix(results_featsets$kcorr_deconf, nrow=10), na.rm=TRUE)
kfold_feats$kcod <- colMeans(matrix(results_featsets$kcod, nrow=10), na.rm=TRUE)
kfold_feats$kcod_deconf <- colMeans(matrix(results_featsets$kcod_deconf, nrow=10), na.rm=TRUE)
kfold_feats$knmae <- colMeans(matrix(results_featsets$knmae, nrow=10), na.rm=TRUE)
kfold_feats$knmae_deconf <- colMeans(matrix(results_featsets$knmae_deconf, nrow=10), na.rm=TRUE)

# get max. NMAE instead of mean across folds:
n <- nrow(results_featsets)
g <- rep(1:n, each = 10, length = n)
kfold_feats$knmae2 <- tapply(results_featsets$knmae, g, max)
kfold_feats$knmae_deconf2 <- tapply(results_featsets$knmae_deconf, g, max)


###### CV (HMM training scheme) results #####
# adjust variable types, classes, and orders
results_CV$features <- factor(
  results_CV$features, 
  levels=c('Fisher', 'naive', 'naive_norm'))
results_CV$kernel <- factor(results_CV$kernel, levels=c('linear','Gaussian'))
results_CV$varN2 <- as.factor(results_CV$varN)
results_CV$varN3 <- varN3[results_CV$varN] # actual variable names

results_CV$model <- results_CV$features:results_CV$kernel
results_CV$model <- droplevels(results_CV$model)
results_CV$model <- factor(results_CV$model, levels = c(
  "Fisher:linear", "naive:linear", "naive_norm:linear",
  "Fisher:Gaussian", "naive:Gaussian", "naive_norm:Gaussian"))

results_CV$modelvar <- results_CV$model:results_CV$varN2 # for grouping by method and target variable

results_CV$training <- factor(results_CV$training,
                              levels = c("tog", "sep"))
results_CV$fullname <- results_CV$model:results_CV$training


###### Outlier removal (main results) #######
# remove most extreme outliers
ggplot(results_main, aes(x=kcod_deconf, fill=model)) + 
  geom_histogram(alpha=0.2, position = 'identity', bins=100) + 
  labs(fill="") + xlim(c(0, 0.6))
# for kcod_deconf
boxplot(results_main$kcod_deconf)$out
Q <- quantile(results_main$kcod_deconf, probs=c(.01, .99), na.rm = TRUE)
iqr <- abs(Q[1]-Q[2])
up <- Q[2]+1.5*iqr
low <- Q[1]-1.5*iqr
results_wo_out_kcod <- subset(results_main, kcod_deconf > low & kcod_deconf < up)
boxplot(results_wo_out_kcod$kcod_deconf)

ggplot(results_wo_out_kcod, aes(x=kcod_deconf, fill=model)) + 
  geom_histogram(alpha=0.2, position = 'identity', bins=100) + 
  labs(fill="") #+ xlim(c(0, 0.6))

# Not possible/relevant for NMAE because these are extreme value distributions


###### Reliability (main results) ######
## Updated summary & robustness tables:

# descriptive stats for model performance measures (minimum, mean, median, and maximum for correlation coefficient, R-squared and NMAE)
corr_min <- tapply(results_main$kcorr_deconf, results_main$model, min, na.rm=TRUE)
corr_mean <- tapply(results_main$kcorr_deconf, results_main$model, mean, na.rm=TRUE)
corr_median <- tapply(results_main$kcorr_deconf, results_main$model, median, na.rm=TRUE)
corr_max <- tapply(results_main$kcorr_deconf, results_main$model, max, na.rm=TRUE)
Rsq_min <- tapply(results_main$kcod_deconf, results_main$model, min, na.rm=TRUE)
Rsq_mean <- tapply(results_main$kcod_deconf, results_main$model, mean, na.rm=TRUE)
Rsq_median <- tapply(results_main$kcod_deconf, results_main$model, median, na.rm=TRUE)
Rsq_max <- tapply(results_main$kcod_deconf, results_main$model, max, na.rm=TRUE)

maxerr_min <- tapply(results_main$knmae, results_main$model, min, na.rm=TRUE)
maxerr_mean <- tapply(results_main$knmae, results_main$model, mean, na.rm=TRUE)
maxerr_median <- tapply(results_main$knmae, results_main$model, median, na.rm=TRUE)
maxerr_max <- tapply(results_main$knmae, results_main$model, max, na.rm=TRUE)

short_df <- data.frame(corr_min, corr_mean, corr_median, corr_max, Rsq_min, Rsq_mean, Rsq_median, Rsq_max, maxerr_min, maxerr_mean, maxerr_median, maxerr_max)
write.csv(short_df, file="MAINSummary.csv")

# Robustness:
# standard deviation across folds & iterations (grouped by method and target variable)
corr_sd <- tapply(results_main$kcorr_deconf, results_main$modelvar, sd, na.rm=TRUE) # correlation coefficient
cod_sd <- tapply(results_main$kcod_deconf, results_main$modelvar, sd, na.rm=TRUE) # "-" R-squared
nmae_sd <- tapply(results_main$knmae, results_main$modelvar, sd, na.rm=TRUE) # "-" NMAE
# best-predicted target variables by method:
corr_sd_mean <- tapply(results_main$kcorr_deconf, results_main$modelvar, mean, na.rm=TRUE)
sort(corr_sd_mean)

# assemble standard deviations into data frame
robust_dftmp <- data.frame()
robust_dftmp[1:35,1] <- 'Fisher_lin'
robust_dftmp[36:70,1] <- 'Naive_lin'
robust_dftmp[71:105,1] <- 'Naivenorm_lin'
robust_dftmp[106:140,1] <- 'Fisher_gaus'
robust_dftmp[141:175,1] <- 'Naive_gaus'
robust_dftmp[176:210,1] <- 'Naivenorm_gaus'
robust_dftmp[211:245,1] <- 'KL'
robust_dftmp[246:280,1] <- 'KL_ta'
robust_dftmp[281:315,1] <- 'Fro'
robust_dftmp[316:350,1] <- 'statFC_RR'
robust_dftmp[351:385,1] <- 'statFC_RRRiem'
robust_dftmp[386:420,1] <- 'statFC_EN'
robust_dftmp[421:455,1] <- 'statFC_ENRiem'
robust_dftmp[456:490,1] <- 'SelectedEdges'
robust_dftmp[,2] <- corr_sd
robust_dftmp[,3] <- cod_sd
robust_dftmp[,4] <- nmae_sd

robust_df <- data.frame()
robust_df[1:14,1] <- tapply(robust_dftmp[,2], robust_dftmp[,1], mean)
robust_df[1:14,2] <- tapply(robust_dftmp[,3], robust_dftmp[,1], mean)
robust_df[1:14,3] <- tapply(robust_dftmp[,4], robust_dftmp[,1.], mean)
# NOTE: This will be in alphabetic order!
write.csv(robust_df, file="MAINRobustness.csv")

# Risk of excessive errors
# Percentage of runs in which NMAE exceeds 10 (low risk), 100 (medium risk) and 1000 (high risk)
sum(results_main[results_main$model=='Fisher:Gaussian',]$knmae > 1)
low_risks <- tapply(results_main$knmae > 10, results_main$model, sum, na.rm=TRUE)/35
medium_risks <- tapply(results_main$knmae > 100, results_main$model, sum, na.rm=TRUE)/35
high_risks <- tapply(results_main$knmae > 1000, results_main$model, sum, na.rm=TRUE)/35



#### Stats ####
###### Main results: Model performance ######
# General formula: repeated k-fold cross-validation correction:
# t = ((1/(k*r))*sum(kcorr_difference(all_k_all_reps)))/(sqrt((1/(k*r))+(ntest/ntrain)*var))
# var = (1/(k*r-1))*sum((kcorr_difference(all_k_all_reps)-(1/(k*r))*sum(kcorr_difference(all_k_all_reps))^2))

ntrain = 901 # number of samples in training set
ntest = 100 # number of samples in test set
k = 10 # number of folds
r = 100 # number of repetitions
nv = 35 # number of variables

outcomes <- "kcorr_deconf"

# compare linear Fisher kernel w. linear naive & naive norm.
# compare Gaussian Fisher kernel w. Gaussian naive, naive norm. & KL divergence
# compare linear Fisher kernel w. static FC methods
# compare static FC in Euclidean space w. static FC in tangent space
comparisons1 <- c("naive:linear", "naive_norm:linear")
comparisons2 <- c("naive:Gaussian", "naive_norm:Gaussian", "KL:Gaussian")
comparisons3 <- c("KL_ta:Gaussian", "Fro:Gaussian","statFC_RR:linear", "statFC_RRRiem:linear", "SelectedEdges:linear")

left = c(rep("Fisher:linear", length(comparisons1)), rep("Fisher:Gaussian", length(comparisons2)), "Fisher:linear",
         rep("Fisher:linear", length(comparisons3)), "statFC_RR:linear")
right = c(comparisons1, comparisons2, "Fisher:Gaussian", comparisons3, "statFC_RRRiem:linear")

for (j in 1:length(outcomes)) {
  t = c()
  p = c()
  
  data = results_main
  
  for (i in 1:length(left)) {
    diff = subset(data, model==left[i], select=outcomes[j])-
      subset(data, model==right[i], select=outcomes[j])
    var = (1/(k*r*nv-1))*sum((diff-(1/(k*r*nv))*sum(diff))^2)
    t[i] = ((1/(k*r*nv))*sum(diff))/(sqrt((1/(k*r*nv))+(ntest/ntrain)*var)) # repeated k-fold cross-validation corrected t-value
  }
  
  main_stats = data.frame(left, right, t)
}

main_stats$p = pt(-abs(kcorr_deconf_stats2$t), df = k*r*nv-1) # uncorrected p-value
main_stats$p_adj = p.adjust(kcorr_deconf_stats2$p, method = "BH") # Benjamini-Hochberg-corrected p-value

write.csv(main_stats, 'MAINstats.csv', col.names=c("Model", "vs.", "t-value", "p-value (uncorrected)", "p-value (BH)"))


###### Main results: Robustness ######
# since we have averaged over folds and repetitions, S.D. values should only be paired for the 35 target variables

outcomes <- "kcorr_deconf"

# comparisons as above
left <- c(rep("Fisher_lin", 7), rep("Fisher_gaus", 3), "Fisher_lin", "statFC_RR")
right <- c("Naive_lin", "Naivenorm_lin", "KL_ta", "Fro",
           "statFC_RR", "statFC_RRRiem", "SelectedEdges", "Naive_gaus",
           "Naivenorm_gaus", "KL", "Fisher_gaus", "statFC_RRRiem")

for (j in 1:length(outcomes)) {
  t = c()
  p = c()
  
  data = robust_dftmp
  
  for (i in 1:length(left)) {
    pttest <- t.test(subset(robust_dftmp, V1==left[i], select=outcomes[j]),
                     subset(robust_dftmp, V1==right[i], select=outcomes[j],
                            paired=TRUE))
    t[i] = pttest$statistic # t-value
    p[i] = pttest$p.value # uncorrected p-value
  }
  
  robustness_stats = data.frame(left, right, t, p)
}


robustness_stats$p_adj<-p.adjust(robustness_stats$p, method="BH")
print(robustness_stats$p_adj)

write.csv(robustness_stats, 'MAINrobustnessstats.csv', col.names=c("Model", "vs.", "t-value", "p-value (uncorrected)", "p-value (BH)"))


###### Feature sets stats ######
ntrain = 901
ntest = 100
k = 10
r = 100
nv = 35

outcomes <- "kcorr_deconf"
models <- levels(results_featsets$features)

# compare full kernels with all reduced ones
comparisons <- levels(results_featsets$featureset)[2:4]

left = c(rep("Fisher:full", 3), rep("naive:full", 3), rep("naive_norm:full", 3))
right = rep(comparisons, 3)
right[1:3] <- paste0("Fisher:", right[1:3])
right[4:6] <- paste0("naive:", right[4:6])
right[7:9] <- paste0("naive_norm:", right[7:9])

for (j in 1:length(outcomes)) {
  t = c()
  p = c()
  data = results_featsets
  
  for (i in 1:length(left)) {
    diff = subset(data, fullname==left[i], select=outcomes[j])-
      subset(data, fullname==right[i], select=outcomes[j])
    var = (1/(k*r*nv-1))*sum((diff-(1/(k*r*nv))*sum(diff))^2)
    t[i] = ((1/(k*r*nv))*sum(diff))/(sqrt((1/(k*r*nv))+(ntest/ntrain)*var)) # repeated k-fold cross-validation corrected t-value
  }
  
  featsets_stats = data.frame(left, right, t)
}

featsets_stats$p = pt(-abs(featsets_stats$t), df = k*r*nv-1) # uncorrected p-value
featsets_stats$p_adj = p.adjust(featsets_stats$p, method = "BH") # Benjamini-Hochberg-corrected p-value

write.csv(featsets_stats, 'FEATURESETSstats.csv', col.names=c("Model", "vs.", "t-value", "p-value (uncorrected)", "p-value (BH)"))


###### CV stats ######

outcomes <- "kcorr_deconf"

comparisons <- levels(results_CV$training)
models <- unique(results_CV$model)
left <- rep(comparisons[1], 6)
left <- paste0(models, ":", left)
right <- rep(comparisons[2], 6)
right <- paste0(models, ":", right)

for (j in 1:length(outcomes)) {
  t = c()
  p = c()
  data = results_CV
  
  for (i in 1:length(left)) {
    x = subset(data, fullname==left[i], select=outcomes[j])
    y = subset(data, fullname==right[i], select=outcomes[j])
    ktest <- kfold_ttest(x=x, y=y, n=nv, k=k)
    t[i] = ktest$statistic # repeated k-fold cross-validation corrected t-value
    p[i] = ktest$p.value # uncorrected p-value
  }
  
  CV_stats = data.frame(left, right, t, p)
}

CV_stats$p_adj = p.adjust(CV_stats$p, method = "BH") # Benjamini-Hochberg-corrected p-value

write.csv(CV_stats, 'CVstats.csv', col.names=c("Model", "vs.", "t-value", "p-value (uncorrected)", "p-value (BH)"))




#### Figures ####
# plotting averaged performance across 10 folds because scatter plot will otherwise eat up too much memory (~400,000 points per panel)

###### Figure 2: Main results ######

# Fig. 2A: Correlation coefficient
fig2a <- ggplot(kfold_table, aes(kcorr_deconf, model))
fig2a <- fig2a + 
  geom_jitter(aes(fill=model), alpha=0.05, size=0.7, width=0, height=0.4, show.legend=FALSE, stroke=NA, pch =21) +
  geom_violin(fill=NA, draw_quantiles = 0.5, show.legend=FALSE) + theme_classic() + 
  scale_y_discrete(labels=c("Selected Edges", "Elastic Net (Riem.)", "Elastic Net", "Ridge Reg. (Riem.)", "Ridge Reg.", "Log-Euclidean", "KL div. (ta)",
                            "KL div.", "Naive norm. Gaussian", "Naive Gaussian",
                            "Fisher kernel Gaussian", "Naive norm. linear", "Naive linear", "Fisher kernel linear"),
                   limits=rev, 
                   name=NULL) + 
  xlab("Correlation coefficient (r)") 

# Fig 2B: R-squared
fig2b <- ggplot(kfold_table, aes(kcod_deconf, model))
fig2b <- fig2b + 
  geom_jitter(aes(fill=model), alpha=0.05, size=0.7, width=0, height=0.4, show.legend=FALSE, stroke=NA, pch =21) +
  geom_violin(fill=NA, draw_quantiles = 0.5, show.legend=FALSE) + theme_classic() + 
  scale_y_discrete(labels=NULL,
                   limits=rev, 
                   name=NULL) + 
  xlab("Coefficient of determination (R2)") + 
  xlim(c(-0.1,0.35))

# Fig 2C: NMAXAE
fig2c <- ggplot(kfold_table, aes(knmae2, model))
fig2c <- fig2c + 
  geom_jitter(aes(fill=model), alpha=0.05, size=0.7, width=0, height=0.4, show.legend=FALSE, stroke=NA, pch =21) +
  geom_violin(fill=NA, draw_quantiles = 0.5, show.legend=FALSE) + theme_classic() + 
  scale_x_log10("Normalised Maximum Absolute Error (NMAXAE)") + theme(legend.position="none") + 
  scale_y_discrete(labels=c("Selected Edges", "Elastic Net (Riem.)", "Elastic Net", "Ridge Reg. (Riem.)", "Ridge Reg.", "Log-Euclidean", "KL div. (ta)",
                            "KL div.", "Naive norm. Gaussian", "Naive Gaussian",
                            "Fisher kernel Gaussian", "Naive norm. linear", "Naive linear", "Fisher kernel linear"),
                   limits=rev, 
                   name=NULL)

# Fig 2D: Standard deviation of correlation r
fig2d <- ggplot(robust_dftmp, aes(V2, V1))
fig2d <- fig2d + 
  geom_jitter(aes(fill=V1), alpha=1.5, size=0.5, width=0, height=0.2, show.legend=FALSE, stroke=NA, pch =21) +
  geom_violin(fill=NA, draw_quantiles = 0.5, show.legend=FALSE) + theme_classic() + 
  theme(legend.position="none") + 
  scale_y_discrete(limits=c("SelectedEdges", "statFC_ENRiem", "statFC_EN", "statFC_RRRiem", "statFC_RR", "Fro", "KL_ta",
                            "KL", "Naivenorm_gaus", "Naive_gaus", "Fisher_gaus", "Naivenorm_lin", "Naive_lin", "Fisher_lin"),
                   labels=NULL, name=NULL) + xlab("Standard deviation of correlation coefficient") 

fig2 = ggarrange(fig2a, fig2b, fig2c, fig2d,
                 labels = c("A", "B", "C", "D"),
                 ncol = 2,
                 nrow = 2,
                 widths = c(1.5,1),
                 heights = c(1.5,1))

# save figure (if figure too large, consider splitting into panels)
CairoPDF("Figure2.pdf", 7, 6.7, bg="transparent", pointsize=4)
fig2
dev.off()


###### Figure 3: Accuracy by variable ######
# all variables, not ordered
varplot_all <- ggplot(results_main, aes(kcorr_deconf, varN2))
varplot_all <- varplot_all + geom_boxplot(aes(fill=model, colour=model), outlier.size = 0.1) + theme_classic()
varplot_all

# order variables by average prediction accuracy
vartable_tmp <- tapply(kfold_table$kcorr_deconf, list(kfold_table$model, kfold_table$varN2), mean, na.rm=TRUE)
vartable_ordered <- sort(sapply(data.frame(vartable_tmp), mean, na.rm=TRUE), decreasing=TRUE, index.return=TRUE)
vars_high <- vartable_ordered$ix[1:18]
vars_low <- vartable_ordered$ix[19:35]

high_pred <- subset(kfold_table, varN%in%vars_high)
low_pred <- subset(kfold_table, varN%in%vars_low)
high_pred <- droplevels(high_pred)
low_pred <- droplevels(low_pred)

# flip order of models for plotting
high_pred$model2 <- fct_rev(high_pred$model)
low_pred$model2 <- fct_rev(low_pred$model)

fig3left <- ggplot(high_pred, aes(y = reorder(varN3, -kcorr_deconf), x = kcorr_deconf))
fig3left <- fig3left + geom_boxplot(aes(fill=model2, colour=model2), outlier.size = 0.1) + 
  theme_classic() + xlim(min(kfold_table$kcorr_deconf, na.rm=TRUE), max(kfold_table$kcorr_deconf, na.rm=TRUE)) +
  scale_y_discrete(limits=rev) 
fig3left

fig3right <- ggplot(low_pred, aes(y = reorder(varN3, -kcorr_deconf), x = kcorr_deconf))
fig3right <- fig3right + geom_boxplot(aes(fill=model2, colour=model2), outlier.size = 0.1) + 
  theme_classic() + xlim(min(kfold_table$kcorr_deconf, na.rm=TRUE), max(kfold_table$kcorr_deconf, na.rm=TRUE)) + 
  scale_y_discrete(limits=rev) 
fig3right

# save figure
CairoPDF("Figure3_left.pdf", 9, 10, bg="transparent")
fig3left
dev.off()

CairoPDF("Figure3_right.pdf", 9, 10, bg="transparent")
fig3right
dev.off()



###### Figure 4: Simulations Feature sets ######
# Figure 4 contains only simulation results with plots done in Matlab. See SimulateFeatures_main.m for figure script



###### Figure 5: Feature sets #####

# Fig 5A left: Correlation coefficient
fig5aleft <- ggplot(kfold_feats, aes(kcorr_deconf, fullname))
fig5aleft <- fig5aleft + 
  geom_jitter(aes(fill=fullname), alpha=0.05, size=0.7, width=0, height=0.4, show.legend=FALSE, stroke=NA, pch=21) +
  geom_violin(fill=NA, draw_quantiles = 0.5, show.legend=FALSE) + theme_classic() + 
  scale_y_discrete(limits=rev, 
                   name=NULL) + 
  xlab("Correlation coefficient (r)") 
fig5aleft

# Fig 5A right: R-squared
fig5aright <- ggplot(kfold_feats, aes(kcod_deconf, fullname))
fig5aright <- fig5aright + 
  geom_jitter(aes(fill=fullname), alpha=0.05, size=0.7, width=0, height=0.4, show.legend=FALSE, stroke=NA, pch=21) +
  geom_violin(fill=NA, draw_quantiles = 0.5, show.legend=FALSE) + theme_classic() + 
  scale_y_discrete(labels=NULL,
                   limits=rev, 
                   name=NULL) + 
  xlim(0, max(kfold_feats$kcod_deconf, na.rm=TRUE)) + 
  xlab("Coefficient of determination (R2)")
fig5aright

# Fig 5B: Accuracy by target variable
# for plot by variable, use order calculated above (average prediction accuracy across methods)
high_pred_featsets <- subset(kfold_feats, varN%in%vars_high)
low_pred_featsets <- subset(kfold_feats, varN%in%vars_low)
high_pred_featsets <- droplevels(high_pred_featsets)
low_pred_featsets <- droplevels(low_pred_featsets)

# flip order of models for plotting
high_pred_featsets$model2 <- fct_rev(high_pred_featsets$fullname)
low_pred_featsets$model2 <- fct_rev(low_pred_featsets$fullname)

levels_higheracc <- levels(reorder(high_pred$varN3, -high_pred$kcorr_deconf))
levels_loweracc <- levels(reorder(low_pred$varN3, -low_pred$kcorr_deconf))
high_pred_featsets$varN4 <- factor(high_pred_featsets$varN3, levels = levels_higheracc)
low_pred_featsets$varN4 <- factor(low_pred_featsets$varN3, levels = levels_loweracc)


fig5bleft <- ggplot(high_pred_featsets, aes(y = varN4, x = kcorr_deconf))
fig5bleft <- fig5bleft + geom_boxplot(aes(fill=model2, colour=model2), outlier.size = 0.1) + 
  theme_classic() + xlim(min(kfold_feats$kcorr_deconf), max(kfold_feats$kcorr_deconf)) +
  scale_y_discrete(limits=rev) + theme(legend.position="none")
fig5bleft

fig5bright <- ggplot(low_pred_featsets, aes(y = varN4, x = kcorr_deconf))
fig5bright <- fig5bright + geom_boxplot(aes(fill=model2, colour=model2), outlier.size = 0.1) + 
  theme_classic() + xlim(min(kfold_feats$kcorr_deconf), max(kfold_feats$kcorr_deconf)) + 
  scale_y_discrete(limits=rev) + theme(legend.position="none")
fig5bright

fig5top = ggarrange(fig5aleft, fig5aright,
                 labels = c("A", "A"),
                 ncol = 2,
                 nrow = 1,
                 widths = c(1.5,1))

fig5bottom = ggarrange(fig5bleft, fig5bright,
                    labels = c("B", "B"),
                    ncol = 2,
                    nrow = 1,
                    widths = c(1,1))

# save figure
CairoPDF("Figure5_top.pdf", 7, 2.2, bg="transparent", pointsize=4)
fig5top
dev.off()

CairoPDF("Figure5_bottom.pdf", 7,4.4, bg="transparent", pointsize=4)
fig5bottom
dev.off()



###### Figure 6: CV (HMM training scheme) ######

# Figure 6A left: Correlation coefficient
fig6aleft <- ggplot(results_CV, aes(kcorr_deconf, fullname))
fig6aleft <- fig6aleft + 
  geom_jitter(aes(fill=fullname), alpha=0.3, size=0.3, width=0, height=0.4, show.legend=FALSE, stroke=NA, pch=21) +
  geom_violin(fill=NA, draw_quantiles = 0.5, show.legend=FALSE) + theme_classic() + 
  scale_y_discrete(limits=rev, 
                   name=NULL) + 
  xlab("Correlation coefficient (r)") 
fig6aleft


# Fig 6A right: R-squared
fig6aright <- ggplot(results_CV, aes(kcod_deconf, fullname))
fig6aright <- fig6aright + 
  geom_jitter(aes(fill=fullname), alpha=0.3, size=0.3, width=0, height=0.4, show.legend=FALSE, stroke=NA, pch=21) +
  geom_violin(fill=NA, draw_quantiles = 0.5, show.legend=FALSE) + theme_classic() + 
  scale_y_discrete(labels=NULL,
                   limits=rev, 
                   name=NULL) + 
  xlim(0, max(results_CV$kcod_deconf, na.rm=TRUE)) + 
  xlab("Coefficient of determination (R2)")
fig6aright

# save figure
fig6top = ggarrange(fig6aleft, fig6aright,
                    labels = c("A", "A"),
                    ncol = 2,
                    nrow = 1,
                    widths = c(1.5,1))

CairoPDF("Figure6_top.pdf", 7, 2.2, bg="transparent", pointsize=4)
fig6top
dev.off()


# Figure 6B are simulation results plotted in Matlab. See SimulateCV_main.m for figure scripts

