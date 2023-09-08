###############################################
#          SOURCE CODE FOR CHAPTER 3          #
###############################################

# INSTALLING AND LOADING PACKAGES ----
install.packages("mlr", dependencies = TRUE) # could take several minutes
# only needed once on any R installation

library(mlr)

library(tidyverse)

# LOADING DIABETES DATA ----
data(diabetes, package = "mclust")

diabetesTib <- as_tibble(diabetes)

summary(diabetesTib)

diabetesTib

# PLOT THE RELATIONSHIPS IN THE DATA ----
ggplot(diabetesTib, aes(glucose, insulin, col = class)) +
  geom_point()  +
  theme_bw()

ggplot(diabetesTib, aes(sspg, insulin, col = class)) +
  geom_point() +
  theme_bw()

ggplot(diabetesTib, aes(sspg, glucose, col = class)) +
  geom_point() +
  theme_bw()

# DEFINING THE DIABETES TASK ----
diabetesTask <- makeClassifTask(data = diabetesTib, target = "class")

diabetesTask
# Supervised task: diabetesTib
# Type: classif
# Target: class
# Observations: 145
# Features:
#   numerics     factors     ordered functionals
# 3           0           0           0
# Missings: FALSE
# Has weights: FALSE
# Has blocking: FALSE
# Has coordinates: FALSE
# Classes: 3
# Chemical   Normal    Overt
# 36       76       33
# Positive class: NA

# DEFINING THE KNN LEARNER ----
knn <- makeLearner("classif.knn", par.vals = list("k" = 2))
knn
# Learner classif.knn from package class
# Type: classif
# Name: k-Nearest Neighbor; Short name: knn
# Class: classif.knn
# Properties: twoclass,multiclass,numerics
# Predict-Type: response
# Hyperparameters: k=2

# LISTING ALL OF MLR'S LEARNERS 当前有152种学习器
listLearners()$class
# [1] "classif.ada"
# [2] "classif.adaboostm1"
# [3] "classif.bartMachine"
# [4] "classif.binomial"
# [5] "classif.boosting"
# [6] "classif.bst"
# [7] "classif.C50"
# [8] "classif.cforest"
# [9] "classif.clusterSVM"
# [10] "classif.ctree"
# [11] "classif.cvglmnet"
# [12] "classif.dbnDNN"
# [13] "classif.dcSVM"
# [14] "classif.earth"
# [15] "classif.evtree"
# [16] "classif.fdausc.glm"
# [17] "classif.fdausc.kernel"
# [18] "classif.fdausc.knn"
# [19] "classif.fdausc.np"
# [20] "classif.FDboost"
# [21] "classif.featureless"
# [22] "classif.fgam"
# [23] "classif.fnn"
# [24] "classif.gamboost"
# [25] "classif.gaterSVM"
# [26] "classif.gausspr"
# [27] "classif.gbm"
# [28] "classif.glmboost"
# [29] "classif.glmnet"
# [30] "classif.h2o.deeplearning"
# [31] "classif.h2o.gbm"
# [32] "classif.h2o.glm"
# [33] "classif.h2o.randomForest"
# [34] "classif.IBk"
# [35] "classif.J48"
# [36] "classif.JRip"
# [37] "classif.kknn"
# [38] "classif.knn"
# [39] "classif.ksvm"
# [40] "classif.lda"
# [41] "classif.LiblineaRL1L2SVC"
# [42] "classif.LiblineaRL1LogReg"
# [43] "classif.LiblineaRL2L1SVC"
# [44] "classif.LiblineaRL2LogReg"
# [45] "classif.LiblineaRL2SVC"
# [46] "classif.LiblineaRMultiClassSVC"
# [47] "classif.logreg"
# [48] "classif.lssvm"
# [49] "classif.lvq1"
# [50] "classif.mda"
# [51] "classif.mlp"
# [52] "classif.multinom"
# [53] "classif.naiveBayes"
# [54] "classif.neuralnet"
# [55] "classif.nnet"
# [56] "classif.nnTrain"
# [57] "classif.OneR"
# [58] "classif.pamr"
# [59] "classif.PART"
# [60] "classif.penalized"
# [61] "classif.plr"
# [62] "classif.plsdaCaret"
# [63] "classif.probit"
# [64] "classif.qda"
# [65] "classif.randomForest"
# [66] "classif.ranger"
# [67] "classif.rda"
# [68] "classif.rFerns"
# [69] "classif.rotationForest"
# [70] "classif.rpart"
# [71] "classif.RRF"
# [72] "classif.saeDNN"
# [73] "classif.sda"
# [74] "classif.sparseLDA"
# [75] "classif.svm"
# [76] "classif.xgboost"
# [77] "cluster.cmeans"
# [78] "cluster.Cobweb"
# [79] "cluster.dbscan"
# [80] "cluster.EM"
# [81] "cluster.FarthestFirst"
# [82] "cluster.kkmeans"
# [83] "cluster.kmeans"
# [84] "cluster.MiniBatchKmeans"
# [85] "cluster.SimpleKMeans"
# [86] "cluster.XMeans"
# [87] "multilabel.cforest"
# [88] "multilabel.rFerns"
# [89] "regr.bartMachine"
# [90] "regr.bcart"
# [91] "regr.bgp"
# [92] "regr.bgpllm"
# [93] "regr.blm"
# [94] "regr.brnn"
# [95] "regr.bst"
# [96] "regr.btgp"
# [97] "regr.btgpllm"
# [98] "regr.btlm"
# [99] "regr.cforest"
# [100] "regr.crs"
# [101] "regr.ctree"
# [102] "regr.cubist"
# [103] "regr.cvglmnet"
# [104] "regr.earth"
# [105] "regr.evtree"
# [106] "regr.FDboost"
# [107] "regr.featureless"
# [108] "regr.fgam"
# [109] "regr.fnn"
# [110] "regr.frbs"
# [111] "regr.gamboost"
# [112] "regr.gausspr"
# [113] "regr.gbm"
# [114] "regr.glm"
# [115] "regr.glmboost"
# [116] "regr.glmnet"
# [117] "regr.GPfit"
# [118] "regr.h2o.deeplearning"
# [119] "regr.h2o.gbm"
# [120] "regr.h2o.glm"
# [121] "regr.h2o.randomForest"
# [122] "regr.IBk"
# [123] "regr.kknn"
# [124] "regr.km"
# [125] "regr.ksvm"
# [126] "regr.laGP"
# [127] "regr.LiblineaRL2L1SVR"
# [128] "regr.LiblineaRL2L2SVR"
# [129] "regr.lm"
# [130] "regr.mars"
# [131] "regr.mob"
# [132] "regr.nnet"
# [133] "regr.pcr"
# [134] "regr.penalized"
# [135] "regr.plsr"
# [136] "regr.randomForest"
# [137] "regr.ranger"
# [138] "regr.rpart"
# [139] "regr.RRF"
# [140] "regr.rsm"
# [141] "regr.rvm"
# [142] "regr.svm"
# [143] "regr.xgboost"
# [144] "surv.cforest"
# [145] "surv.coxph"
# [146] "surv.cvglmnet"
# [147] "surv.gamboost"
# [148] "surv.gbm"
# [149] "surv.glmboost"
# [150] "surv.glmnet"
# [151] "surv.ranger"
# [152] "surv.rpart"

# or list them by function:
listLearners("classif")$class
# [1] "classif.ada"
# [2] "classif.adaboostm1"
# [3] "classif.bartMachine"
# [4] "classif.binomial"
# [5] "classif.boosting"
# [6] "classif.bst"
# [7] "classif.C50"
# [8] "classif.cforest"
# [9] "classif.clusterSVM"
# [10] "classif.ctree"
# [11] "classif.cvglmnet"
# [12] "classif.dbnDNN"
# [13] "classif.dcSVM"
# [14] "classif.earth"
# [15] "classif.evtree"
# [16] "classif.fdausc.glm"
# [17] "classif.fdausc.kernel"
# [18] "classif.fdausc.knn"
# [19] "classif.fdausc.np"
# [20] "classif.FDboost"
# [21] "classif.featureless"
# [22] "classif.fgam"
# [23] "classif.fnn"
# [24] "classif.gamboost"
# [25] "classif.gaterSVM"
# [26] "classif.gausspr"
# [27] "classif.gbm"
# [28] "classif.glmboost"
# [29] "classif.glmnet"
# [30] "classif.h2o.deeplearning"
# [31] "classif.h2o.gbm"
# [32] "classif.h2o.glm"
# [33] "classif.h2o.randomForest"
# [34] "classif.IBk"
# [35] "classif.J48"
# [36] "classif.JRip"
# [37] "classif.kknn"
# [38] "classif.knn"
# [39] "classif.ksvm"
# [40] "classif.lda"
# [41] "classif.LiblineaRL1L2SVC"
# [42] "classif.LiblineaRL1LogReg"
# [43] "classif.LiblineaRL2L1SVC"
# [44] "classif.LiblineaRL2LogReg"
# [45] "classif.LiblineaRL2SVC"
# [46] "classif.LiblineaRMultiClassSVC"
# [47] "classif.logreg"
# [48] "classif.lssvm"
# [49] "classif.lvq1"
# [50] "classif.mda"
# [51] "classif.mlp"
# [52] "classif.multinom"
# [53] "classif.naiveBayes"
# [54] "classif.neuralnet"
# [55] "classif.nnet"
# [56] "classif.nnTrain"
# [57] "classif.OneR"
# [58] "classif.pamr"
# [59] "classif.PART"
# [60] "classif.penalized"
# [61] "classif.plr"
# [62] "classif.plsdaCaret"
# [63] "classif.probit"
# [64] "classif.qda"
# [65] "classif.randomForest"
# [66] "classif.ranger"
# [67] "classif.rda"
# [68] "classif.rFerns"
# [69] "classif.rotationForest"
# [70] "classif.rpart"
# [71] "classif.RRF"
# [72] "classif.saeDNN"
# [73] "classif.sda"
# [74] "classif.sparseLDA"
# [75] "classif.svm"
# [76] "classif.xgboost"
listLearners("regr")$class
# [1] "regr.bartMachine"      "regr.bcart"
# [3] "regr.bgp"              "regr.bgpllm"
# [5] "regr.blm"              "regr.brnn"
# [7] "regr.bst"              "regr.btgp"
# [9] "regr.btgpllm"          "regr.btlm"
# [11] "regr.cforest"          "regr.crs"
# [13] "regr.ctree"            "regr.cubist"
# [15] "regr.cvglmnet"         "regr.earth"
# [17] "regr.evtree"           "regr.FDboost"
# [19] "regr.featureless"      "regr.fgam"
# [21] "regr.fnn"              "regr.frbs"
# [23] "regr.gamboost"         "regr.gausspr"
# [25] "regr.gbm"              "regr.glm"
# [27] "regr.glmboost"         "regr.glmnet"
# [29] "regr.GPfit"            "regr.h2o.deeplearning"
# [31] "regr.h2o.gbm"          "regr.h2o.glm"
# [33] "regr.h2o.randomForest" "regr.IBk"
# [35] "regr.kknn"             "regr.km"
# [37] "regr.ksvm"             "regr.laGP"
# [39] "regr.LiblineaRL2L1SVR" "regr.LiblineaRL2L2SVR"
# [41] "regr.lm"               "regr.mars"
# [43] "regr.mob"              "regr.nnet"
# [45] "regr.pcr"              "regr.penalized"
# [47] "regr.plsr"             "regr.randomForest"
# [49] "regr.ranger"           "regr.rpart"
# [51] "regr.RRF"              "regr.rsm"
# [53] "regr.rvm"              "regr.svm"
# [55] "regr.xgboost"
listLearners("cluster")$class
# [1] "cluster.cmeans"          "cluster.Cobweb"
# [3] "cluster.dbscan"          "cluster.EM"
# [5] "cluster.FarthestFirst"   "cluster.kkmeans"
# [7] "cluster.kmeans"          "cluster.MiniBatchKmeans"
# [9] "cluster.SimpleKMeans"    "cluster.XMeans"

# DEFINE MODEL ----
knnModel <- train(knn, diabetesTask)
knnModel
# Model for learner.id=classif.knn; learner.class=classif.knn
# Trained on: task.id = diabetesTib; obs = 145; features = 3
# Hyperparameters: k=2

# TESTING PERFORMANCE ON TRAINING DATA (VERY BAD PRACTICE) ----
knnPred <- predict(knnModel, newdata = diabetesTib)
knnPred
# Prediction: 145 observations
# predict.type: response
# threshold:
#   time: 0.00
# truth response
# 1 Normal   Normal
# 2 Normal   Normal
# 3 Normal   Normal
# 4 Normal   Normal
# 5 Normal   Normal
# 6 Normal   Normal
# ... (#rows: 145, #cols: 2)

performance(knnPred, measures = list(mmce, acc))
# mmce        acc
# 0.04137931 0.95862069
# https://mlr.mlr-org.com/articles/tutorial/performance.html

# Get the default measure for a task type, task, task description or a learner. Currently these are:
# classif: mmce
# regr: mse
# cluster: db
# surv: cindex
# costsen: mcp
# multilabel: multilabel.hamloss

# PERFORMING HOLD-OUT CROSS-VALIDATION ----
holdout <- makeResampleDesc(method = "Holdout",
                            split = 2/3,
                            stratify = TRUE)
holdout
# Resample description: holdout with 0.67 split rate.
# Predict: test
# Stratification: TRUE

holdoutCV <- resample(learner = knn,
                      task = diabetesTask,
                      resampling = holdout,
                      measures = list(mmce, acc))

# Resampling: holdout
# Measures:             mmce      acc
# [Resample] iter 1:    0.0612245 0.9387755
#
#
# Aggregated Result: mmce.test.mean=0.0612245,acc.test.mean=0.9387755
holdoutCV$aggr
# mmce.test.mean  acc.test.mean
# 0.06122449     0.93877551

calculateConfusionMatrix(holdoutCV$pred, relative = TRUE)
# Relative confusion matrix (normalized by row/column):
#   predicted
# true       Chemical  Normal    Overt     -err.-
# Chemical 0.83/0.91 0.17/0.07 0.00/0.00 0.17
# Normal   0.04/0.09 0.96/0.93 0.00/0.00 0.04
# Overt    0.00/0.00 0.00/0.00 1.00/1.00 0.00
# -err.-        0.09      0.07      0.00 0.06


# Absolute confusion matrix:
#   predicted
# true       Chemical Normal Overt -err.-
# Chemical       10      2     0      2
# Normal          1     25     0      1
# Overt           0      0    11      0
# -err.-          1      2     0      3

# PERFORMING REPEATED K-FOLD CROSS-VALIDATION ----
kFold <- makeResampleDesc(method = "RepCV",
                          folds = 10,
                          reps = 50,
                          stratify = TRUE)

kFoldCV <- resample(learner = knn, task = diabetesTask,
                    resampling = kFold, measures = list(mmce, acc))

kFoldCV$aggr
# mmce.test.mean  acc.test.mean
# 0.1022691      0.8977309
kFoldCV$measures.test

calculateConfusionMatrix(kFoldCV$pred, relative = TRUE)
# Relative confusion matrix (normalized by row/column):
#   predicted
# true       Chemical  Normal    Overt     -err.-
# Chemical 0.82/0.78 0.10/0.05 0.08/0.10 0.18
# Normal   0.04/0.08 0.96/0.95 0.00/0.00 0.04
# Overt    0.16/0.14 0.00/0.00 0.84/0.90 0.16
# -err.-        0.22      0.05      0.10 0.10
#
#
# Absolute confusion matrix:
#   predicted
# true       Chemical Normal Overt -err.-
# Chemical     1470    178   152    330
# Normal        143   3657     0    143
# Overt         267      0  1383    267
# -err.-        410    178   152    740

# PERFORMING LEAVE-ONE-OUT CROSS-VALIDATION ----
LOO <- makeResampleDesc(method = "LOO")
LOO
# Resample description: LOO with NA iterations.
# Predict: test
# Stratification: FALSE

LOOCV <- resample(learner = knn,
                  task = diabetesTask, resampling = LOO,
                  measures = list(mmce, acc))

LOOCV$aggr
# mmce.test.mean  acc.test.mean
# 0.1034483      0.8965517

calculateConfusionMatrix(LOOCV$pred, relative = TRUE)
# Relative confusion matrix (normalized by row/column):
#   predicted
# true       Chemical  Normal    Overt     -err.-
# Chemical 0.81/0.78 0.11/0.05 0.08/0.10 0.19
# Normal   0.03/0.05 0.97/0.95 0.00/0.00 0.03
# Overt    0.18/0.16 0.00/0.00 0.82/0.90 0.18
# -err.-        0.22      0.05      0.10 0.10


# Absolute confusion matrix:
#   predicted
# true       Chemical Normal Overt -err.-
# Chemical       29      4     3      7
# Normal          2     74     0      2
# Overt           6      0    27      6
# -err.-          8      4     3     15

# HYPERPARAMETER TUNING OF K ----
knnParamSpace <- makeParamSet(makeDiscreteParam("k", values = 1:10))

gridSearch <- makeTuneControlGrid()

cvForTuning <- makeResampleDesc("RepCV", folds = 10, reps = 20)

tunedK <- tuneParams("classif.knn", task = diabetesTask,
                     resampling = cvForTuning,
                     par.set = knnParamSpace,
                     control = gridSearch)

tunedK
# Tune result:
#   Op. pars: k=7
# mmce.test.mean=0.0791905
tunedK$x
# $k
# [1] 7

knnTuningData <- generateHyperParsEffectData(tunedK)
# HyperParsEffectData:
# Hyperparameters: k
# Measures: mmce.test.mean
# Optimizer: TuneControlGrid
# Nested CV Used: FALSE
# Snapshot of data:
#   k mmce.test.mean iteration exec.time
# 1 1     0.10995238         1     0.442
# 2 2     0.10442857         2     0.313
# 3 3     0.09111905         3     0.672
# 4 4     0.09100000         4     0.302
# 5 5     0.08204762         5     0.326
# 6 6     0.08411905         6     0.307
plotHyperParsEffect(knnTuningData, x = "k", y = "mmce.test.mean",
                    plot.type = "line") +
                    theme_bw()

# TRAINING FINAL MODEL WITH TUNED K ----
tunedKnn <- setHyperPars(makeLearner("classif.knn"), par.vals = tunedK$x)

tunedKnnModel <- train(tunedKnn, diabetesTask)

# INCLUDING HYPERPARAMETER TUNING INSIDE NESTED CROSS-VALIDATION ----
inner <- makeResampleDesc("CV")

outer <- makeResampleDesc("RepCV", folds = 10, reps = 5)

knnWrapper <- makeTuneWrapper("classif.knn", resampling = inner,
                              par.set = knnParamSpace,
                              control = gridSearch)

cvWithTuning <- resample(knnWrapper, diabetesTask, resampling = outer)
# Aggregated Result: mmce.test.mean=0.0920000

cvWithTuning
# Resample Result
# Task: diabetesTib
# Learner: classif.knn.tuned
# Aggr perf: mmce.test.mean=0.0920000
# Runtime: 12.1647

# USING THE MODEL TO MAKE PREDICTIONS ----
newDiabetesPatients <- tibble(glucose = c(82, 108, 300),
                              insulin = c(361, 288, 1052),
                              sspg = c(200, 186, 135))

newDiabetesPatients
# A tibble: 3 × 3
# glucose insulin  sspg
# <dbl>   <dbl> <dbl>
# 1      82     361   200
# 2     108     288   186
# 3     300    1052   135
newPatientsPred <- predict(tunedKnnModel,
                           newdata = newDiabetesPatients)
newPatientsPred
# Prediction: 3 observations
# predict.type: response
# threshold:
#   time: 0.00
# response
# 1   Normal
# 2   Normal
# 3    Overt
getPredictionResponse(newPatientsPred)
# [1] Normal Normal Overt
# Levels: Chemical Normal Overt

# EXERCISES ----
# 1
ggplot(diabetesTib, aes(glucose, insulin,
                        shape = class)) +
  geom_point()  +
  theme_bw()

ggplot(diabetesTib, aes(glucose, insulin,
                        shape = class, col = class)) +
  geom_point()  +
  theme_bw()

# 2
holdoutNoStrat <- makeResampleDesc(method = "Holdout", split = 0.9,
                            stratify = FALSE)

# 3
kFold500 <- makeResampleDesc(method = "RepCV", folds = 3, reps = 500,
                          stratify = TRUE)

kFoldCV500 <- resample(learner = knn, task = diabetesTask,
                    resampling = kFold500, measures = list(mmce, acc))

kFold5 <- makeResampleDesc(method = "RepCV", folds = 3, reps = 5,
                             stratify = TRUE)

kFoldCV5 <- resample(learner = knn, task = diabetesTask,
                       resampling = kFold5, measures = list(mmce, acc))

kFoldCV500$aggr
kFoldCV5$aggr

calculateConfusionMatrix(kFoldCV$pred, relative = TRUE)

# 4
makeResampleDesc(method = "LOO", stratify = TRUE)

makeResampleDesc(method = "LOO", reps = 5)

# both will result in an error as LOO cross-validation cannot
# be stratified or repeated

# 5
data(iris)

irisTask <- makeClassifTask(data = iris, target = "Species")

knnParamSpace <- makeParamSet(makeDiscreteParam("k", values = 1:25))

gridSearch <- makeTuneControlGrid()

cvForTuning <- makeResampleDesc("RepCV", folds = 10, reps = 20)

tunedK <- tuneParams("classif.knn", task = irisTask,
                     resampling = cvForTuning,
                     par.set = knnParamSpace,
                     control = gridSearch)

tunedK
# Tune result:
#   Op. pars: k=14
# mmce.test.mean=0.0260000
tunedK$x
# $k
# [1] 14
knnTuningData <- generateHyperParsEffectData(tunedK)
knnTuningData
# HyperParsEffectData:
#   Hyperparameters: k
# Measures: mmce.test.mean
# Optimizer: TuneControlGrid
# Nested CV Used: FALSE
# Snapshot of data:
#   k mmce.test.mean iteration exec.time
# 1 1     0.04133333         1     0.344
# 2 2     0.04833333         2     0.327
# 3 3     0.04033333         3     0.331
# 4 4     0.03733333         4     0.313
# 5 5     0.03533333         5     0.317
# 6 6     0.03400000         6     0.338
plotHyperParsEffect(knnTuningData, x = "k", y = "mmce.test.mean",
                    plot.type = "line") +
                    theme_bw()

tunedKnn <- setHyperPars(makeLearner("classif.knn"), par.vals = tunedK$x)

tunedKnnModel <- train(tunedKnn, irisTask)
tunedKnnModel
# Model for learner.id=classif.knn; learner.class=classif.knn
# Trained on: task.id = iris; obs = 150; features = 4
# Hyperparameters: k=14

# 6
inner <- makeResampleDesc("CV")

outerHoldout <- makeResampleDesc("Holdout", split = 2/3, stratify = TRUE)

knnWrapper <- makeTuneWrapper("classif.knn", resampling = inner,
                              par.set = knnParamSpace,
                              control = gridSearch)

holdoutCVWithTuning <- resample(knnWrapper, irisTask,
                                resampling = outerHoldout)

holdoutCVWithTuning
# Resample Result
# Task: iris
# Learner: classif.knn.tuned
# Aggr perf: mmce.test.mean=0.0392157
# Runtime: 0.742559

# 7
outerKfold <- makeResampleDesc("CV", iters = 5, stratify = TRUE)

kFoldCVWithTuning <- resample(knnWrapper, irisTask,
                              resampling = outerKfold)

kFoldCVWithTuning
# Resample Result
# Task: iris
# Learner: classif.knn.tuned
# Aggr perf: mmce.test.mean=0.0266667
# Runtime: 3.33468
resample(knnWrapper, irisTask, resampling = outerKfold)
# Aggregated Result: mmce.test.mean=0.0400000


# Resample Result
# Task: iris
# Learner: classif.knn.tuned
# Aggr perf: mmce.test.mean=0.0400000
# Runtime: 3.48486

# repeat each validation procedure 10 times and save the mmce value
# WARNING: this may take a few minutes to complete

kSamples <- map_dbl(1:10, ~resample(
  knnWrapper, irisTask, resampling = outerKfold)$aggr
)
# Aggregated Result: mmce.test.mean=0.0466667
hSamples <- map_dbl(1:10, ~resample(
  knnWrapper, irisTask, resampling = outerHoldout)$aggr
)
# Aggregated Result: mmce.test.mean=0.0392157
hist(kSamples, xlim = c(0, 0.11))
hist(hSamples, xlim = c(0, 0.11))

# holdout CV gives more variable estimates of model performance

