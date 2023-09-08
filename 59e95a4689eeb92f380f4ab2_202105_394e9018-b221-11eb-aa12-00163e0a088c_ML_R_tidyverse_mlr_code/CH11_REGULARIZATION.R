###############################################
#          SOURCE CODE FOR CHAPTER 11          #
###############################################

# LOAD PACKAGES ----
library(mlr)

library(tidyverse)

# LOAD DATA ----
#install.packages("lasso2")

data(Iowa, package = "lasso2")

iowaTib <- as_tibble(Iowa)

iowaTib
# A tibble: 33 × 10
# Year Rain0 Temp1 Rain1 Temp2 Rain2 Temp3 Rain3 Temp4 Yield
# <int> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#   1  1930  17.8  60.2  5.83  69    1.49  77.9  2.42  74.4  34
# 2  1931  14.8  57.5  3.83  75    2.72  77.2  3.3   72.6  32.9
# 3  1932  28.0  62.3  5.17  72    3.12  75.8  7.1   72.2  43
# 4  1933  16.8  60.5  1.64  77.8  3.45  76.4  3.01  70.5  40
# 5  1934  11.4  69.5  3.49  77.2  3.85  79.7  2.84  73.4  23
# 6  1935  22.7  55    7     65.9  3.35  79.4  2.42  73.6  38.4
# 7  1936  17.9  66.2  2.85  70.1  0.51  83.4  3.48  79.2  20
# 8  1937  23.3  61.8  3.8   69    2.63  75.9  3.99  77.8  44.6
# 9  1938  18.5  59.5  4.67  69.2  4.24  76.5  3.82  75.7  46.3
# 10  1939  18.6  66.4  5.32  71.4  3.15  76.2  4.72  70.7  52.2
# ℹ 23 more rows
# ℹ Use `print(n = ...)` to see more rows

# PLOTTING THE DATA ----
iowaUntidy <- gather(iowaTib, "Variable", "Value", -Yield) # WideType to LongType

rio::export(iowaTib, file = "/Users/xinxingjing/R_Space/Machine_Learning_with_R_tidyverse_mlr/CH11_REGULARIZATION/iowaTib.xlsx")
rio::export(iowaUntidy, file = "/Users/xinxingjing/R_Space/Machine_Learning_with_R_tidyverse_mlr/CH11_REGULARIZATION/iowaUntidy.xlsx")
head(iowaUntidy)
# A tibble: 6 × 3
# Yield Variable Value
# <dbl> <chr>    <dbl>
# 1  34   Year      1930
# 2  32.9 Year      1931
# 3  43   Year      1932
# 4  40   Year      1933
# 5  23   Year      1934
# 6  38.4 Year      1935

summary(iowaUntidy)
# Yield            Variable             Value
# Min.   :20.0   Length:297         Min.   :   0.51
# 1st Qu.:43.1   Class :character   1st Qu.:   5.17
# Median :52.0   Mode  :character   Median :  60.00
# Mean   :50.0                      Mean   : 250.76
# 3rd Qu.:59.9                      3rd Qu.:  73.20
# Max.   :76.0                      Max.   :1962.00

ggplot(iowaUntidy, aes(Value, Yield)) +
  facet_wrap(~ Variable, scales = "free_x") +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw()
#ggsave("Iowa.pdf", width = 10, height = 5)

# MAKE TASK AND RIDGE LEARNER ----
iowaTask <- makeRegrTask(data = iowaTib, target = "Yield")
class(iowaTask)
# [1] "RegrTask"       "SupervisedTask" "Task"

ridge <- makeLearner("regr.glmnet", alpha = 0, id = "ridge")

# FEATURE SELECTION FILTER METHOD ----
filterVals <- generateFilterValuesData(iowaTask)

plotFilterValues(filterVals) + theme_bw()
#ggsave("Filtervals.pdf", width = 10, height = 5)

# TUNING LAMBDA FOR RIDGE ----
# https://stackoverflow.com/questions/50995525/nested-resampling-lasso-regr-cvglment-using-mlr

ridgeParamSpace <- makeParamSet(
  makeNumericParam("s", lower = 0, upper = 15))

randSearch <- makeTuneControlRandom(maxit = 200)
class(randSearch)
# [1] "TuneControlRandom" "TuneControl"       "OptControl"

cvForTuning <- makeResampleDesc("RepCV", folds = 3, reps = 10)
class(cvForTuning)
# [1] "RepCVDesc"    "ResampleDesc"

library(parallel)
library(parallelMap)

parallelStartSocket(cpus = detectCores())

tunedRidgePars <- tuneParams(ridge,
                             task = iowaTask, # ~30 sec
                             resampling = cvForTuning,
                             par.set = ridgeParamSpace,
                             control = randSearch)

class(tunedRidgePars)
# [1] "TuneResult" "OptResult"
parallelStop()

tunedRidgePars
# Tune result:
#   Op. pars: s=5.61
# mse.test.mean=93.8888542

# PLOTTING THE RANDOM SEARCH ----
ridgeTuningData <- generateHyperParsEffectData(tunedRidgePars)

plotHyperParsEffect(ridgeTuningData, x = "s", y = "mse.test.mean",
                    plot.type = "line") +
  theme_bw()

#ggsave("Ridge_lambda.pdf", width = 10, height = 5)

# TRAINING FINAL MODEL WITH TUNED HYPERPARAMETERS ----
tunedRidge <- setHyperPars(ridge, par.vals = tunedRidgePars$x)
class(tunedRidge)
# [1] "regr.glmnet"  "RLearnerRegr" "RLearner"     "Learner"
tunedRidgeModel <- train(tunedRidge, iowaTask)
class(tunedRidgeModel)
# [1] "WrappedModel"

# INTERPRETTING THE RIDGE REGRESSION MODEL ----
ridgeModelData <- getLearnerModel(tunedRidgeModel)
class(ridgeModelData)
# [1] "elnet"  "glmnet"

# plot(ridgeModelData, xvar = "lambda", label = TRUE)
# plot(ridgeModelData, xvar = "norm", label = TRUE)
ridgeCoefs <- coef(ridgeModelData, s = tunedRidgePars$x$s)

ridgeCoefs
# 10 x 1 sparse Matrix of class "dgCMatrix"
# s1
# (Intercept) -933.51089980
# Year           0.54520628
# Rain0          0.35437389
# Temp1         -0.24375777
# Rain1         -0.71510665
# Temp2          0.04260419
# Rain2          1.94334596
# Temp3         -0.57082385
# Rain3          0.64227151
# Temp4         -0.48289549
lmCoefs <- coef(lm(Yield ~ ., data = iowaTib))

lmCoefs
# (Intercept)          Year         Rain0         Temp1         Rain1         Temp2
# -1.637276e+03  8.755817e-01  7.845010e-01 -4.597870e-01 -7.794070e-01  4.822053e-01
# Rain2         Temp3         Rain3         Temp4
# 2.556968e+00  4.890862e-02  4.078031e-01 -6.562937e-01
coefTib <- tibble(Coef = rownames(ridgeCoefs)[-1],
                  Ridge = as.vector(ridgeCoefs)[-1],
                  Lm = as.vector(lmCoefs)[-1])

head(coefTib)
# A tibble: 6 × 3
# Coef    Ridge     Lm
# <chr>   <dbl>  <dbl>
# 1 Year   0.545   0.876
# 2 Rain0  0.354   0.785
# 3 Temp1 -0.244  -0.460
# 4 Rain1 -0.715  -0.779
# 5 Temp2  0.0426  0.482
# 6 Rain2  1.94    2.56
coefUntidy <- gather(coefTib, key = Model, value = Beta, -Coef)
head(coefUntidy)
# A tibble: 6 × 3
# Coef  Model    Beta
# <chr> <chr>   <dbl>
# 1 Year  Ridge  0.545
# 2 Rain0 Ridge  0.354
# 3 Temp1 Ridge -0.244
# 4 Rain1 Ridge -0.715
# 5 Temp2 Ridge  0.0426
# 6 Rain2 Ridge  1.94

summary(coefUntidy)
# Coef              Model                Beta
# Length:18          Length:18          Min.   :-0.7794
# Class :character   Class :character   1st Qu.:-0.4771
# Mode  :character   Mode  :character   Median : 0.2016
# Mean   : 0.2653
# 3rd Qu.: 0.6180
# Max.   : 2.5570

visdat::vis_dat(coefUntidy)

ggplot(coefUntidy, aes(reorder(Coef, Beta), Beta, fill = Model)) +
  geom_bar(stat = "identity", col = "black") +
  facet_wrap(~Model) +
  theme_bw()  +
  theme(legend.position = "none")

#ggsave("Ridge_coefs.pdf", width = 10, height = 5)

# MAKE LASSO LEARNER ----
lasso <- makeLearner("regr.glmnet", alpha = 1, id = "lasso")

# TUNING LAMBDA FOR LASSO ----
lassoParamSpace <- makeParamSet(
  makeNumericParam("s", lower = 0, upper = 15))

parallelStartSocket(cpus = detectCores())

tunedLassoPars <- tuneParams(lasso, task = iowaTask, d# ~30 sec
                             resampling = cvForTuning,
                             par.set = lassoParamSpace,
                             control = randSearch)

parallelStop()

tunedLassoPars
# Tune result:
#   Op. pars: s=1.55
# mse.test.mean=102.1759564

# PLOTTING THE RANDOM SEARCH ----
lassoTuningData <- generateHyperParsEffectData(tunedLassoPars)

plotHyperParsEffect(lassoTuningData, x = "s", y = "mse.test.mean",
                    plot.type = "line") +
  theme_bw()

#ggsave("Lasso_lambda.pdf", width = 10, height = 5)

# TRAINING FINAL MODEL WITH TUNED HYPERPARAMETERS ----
tunedLasso <- setHyperPars(lasso, par.vals = tunedLassoPars$x)

tunedLassoModel <- train(tunedLasso, iowaTask)

# INTERPRETTING THE LASSO REGRESSION MODEL ----
lassoModelData <- getLearnerModel(tunedLassoModel)

# plot(lassoModelData, xvar = "lambda", label = TRUE)
# plot(lassoModelData, xvar = "norm", label = TRUE)
lassoCoefs <- coef(lassoModelData, s = tunedLassoPars$x$s)
#lassoCoefs <- coef(lassoModelData, s = 1.37) #to get the same s as me

lassoCoefs
# 10 x 1 sparse Matrix of class "dgCMatrix"
# s1
# (Intercept) -1.330253e+03
# Year         7.226751e-01
# Rain0        1.745260e-01
# Temp1        .
# Rain1        .
# Temp2        .
# Rain2        1.930557e+00
# Temp3       -6.678213e-02
# Rain3        1.059965e-01
# Temp4       -4.319123e-01
coefTib$LASSO <- as.vector(lassoCoefs)[-1]

coefUntidy <- gather(coefTib, key = Model, value = Beta, -Coef)

ggplot(coefUntidy, aes(reorder(Coef, Beta), Beta, fill = Model)) +
  geom_bar(stat = "identity", col = "black") +
  facet_wrap(~ Model) +
  theme_bw() +
  theme(legend.position = "none")

#ggsave("LASSO_coefs.pdf", width = 10, height = 5)

# MAKE ELASTIC NET LEARNER ----
elastic <- makeLearner("regr.glmnet", id = "elastic")
class(elastic)
# [1] "regr.glmnet"  "RLearnerRegr" "RLearner"     "Learner"

# TUNING LAMBDA AND ALPHA FOR ELASTIC NET ----
elasticParamSpace <- makeParamSet(makeNumericParam("s", lower = 0, upper = 10),
                                  makeNumericParam("alpha", lower = 0, upper = 1))

randSearchElastic <- makeTuneControlRandom(maxit = 400)

parallelStartSocket(cpus = detectCores())

tunedElasticPars <- tuneParams(elastic,
                               task = iowaTask, # ~1 min
                               resampling = cvForTuning,
                               par.set = elasticParamSpace,
                               control = randSearchElastic)

parallelStop()

tunedElasticPars
# Tune result:
#   Op. pars: s=1.21; alpha=0.926
# mse.test.mean=89.3839095
elasticTuningData <- generateHyperParsEffectData(tunedElasticPars)

plotHyperParsEffect(elasticTuningData, x = "s", y = "alpha",
                    z = "mse.test.mean", interpolate = "regr.kknn",
                    plot.type = "heatmap") +
  scale_fill_gradientn(colours = terrain.colors(5)) +
  geom_point(x = tunedElasticPars$x$s, y = tunedElasticPars$x$alpha) +
  theme_bw()

#ggsave("Elastic_tuning.pdf", width = 10, height = 5)

# TRAINING FINAL MODEL WITH TUNED HYPERPARAMETERS ----
tunedElastic <- setHyperPars(elastic, par.vals = tunedElasticPars$x)

tunedElasticModel <- train(tunedElastic, iowaTask)

# INTERPRETTING THE LASSO REGRESSION MODEL ----
elasticModelData <- getLearnerModel(tunedElasticModel)

# plot(elasticModelData, xvar = "lambda", label = TRUE)
# plot(elasticModelData, xvar = "norm", label = TRUE)
elasticCoefs <- coef(elasticModelData, s = tunedElasticPars$x$s)

coefTib$Elastic <- as.vector(elasticCoefs)[-1]

coefUntidy <- gather(coefTib, key = Model, value = Beta, -Coef)
coefUntidy
# A tibble: 36 × 3
# Coef  Model    Beta
# <chr> <chr>   <dbl>
#   1 Year  Ridge  0.545
# 2 Rain0 Ridge  0.354
# 3 Temp1 Ridge -0.244
# 4 Rain1 Ridge -0.715
# 5 Temp2 Ridge  0.0426
# 6 Rain2 Ridge  1.94
# 7 Temp3 Ridge -0.571
# 8 Rain3 Ridge  0.642
# 9 Temp4 Ridge -0.483
# 10 Year  Lm     0.876
# ℹ 26 more rows
# ℹ Use `print(n = ...)` to see more rows
ggplot(coefUntidy, aes(reorder(Coef, Beta), Beta, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", col = "black") +
  facet_wrap(~ Model) +
  theme_bw()

rio::export(coefUntidy, file = "./CH11_REGULARIZATION/coefUntidy_Elastic_LASSO_LM_RIDGE.xlsx")
#ggsave("Elastic_coefs.pdf", width = 10, height = 5)

# BENCHMARKING EACH MODEL BUILDING PROCESS ----
# MAKE TUNING WRAPPERS
ridgeWrapper <- makeTuneWrapper(ridge, resampling = cvForTuning,
                                par.set = ridgeParamSpace,
                                control = randSearch)

lassoWrapper <- makeTuneWrapper(lasso, resampling = cvForTuning,
                                par.set = lassoParamSpace,
                                control = randSearch)

elasticWrapper <- makeTuneWrapper(elastic, resampling = cvForTuning,
                                  par.set = elasticParamSpace,
                                  control = randSearchElastic)

learners = list(ridgeWrapper, lassoWrapper, elasticWrapper, "regr.lm")


library(parallel)
library(parallelMap)

kFold3 <- makeResampleDesc("CV", iters = 3)

parallelStartSocket(cpus = detectCores())

bench <- benchmark(learners, iowaTask, kFold3)

parallelStop()

bench
# task.id    learner.id mse.test.mean
# 1 iowaTib   ridge.tuned      114.3147
# 2 iowaTib   lasso.tuned      124.7072
# 3 iowaTib elastic.tuned      119.7209
# 4 iowaTib       regr.lm      120.5801


# SOLUTIONS TO EXERCISES ----
# 1
ridgeParamSpaceExtended <- makeParamSet(makeNumericParam("s", lower = 0, upper = 50))

parallelStartSocket(cpus = detectCores())

tunedRidgeParsExtended <- tuneParams(ridge, task = iowaTask, # ~30 sec
                                     resampling = cvForTuning,
                                     par.set = ridgeParamSpaceExtended,
                                     control = randSearch)

parallelStop()

ridgeTuningDataExtended <- generateHyperParsEffectData(
                                      tunedRidgeParsExtended)

plotHyperParsEffect(ridgeTuningDataExtended, x = "s", y = "mse.test.mean",
                    plot.type = "line") +
  theme_bw()

# the previous value of s was not just a local minimum,
# but the global minimum

# 2
coefTibInts <- tibble(Coef = rownames(ridgeCoefs),
                      Ridge = as.vector(ridgeCoefs),
                      Lm = as.vector(lmCoefs))

coefUntidyInts <- gather(coefTibInts, key = Model, value = Beta, -Coef)
coefUntidyInts
# A tibble: 20 × 3
# Coef        Model       Beta
# <chr>       <chr>      <dbl>
#   1 (Intercept) Ridge  -934.
# 2 Year        Ridge     0.545
# 3 Rain0       Ridge     0.354
# 4 Temp1       Ridge    -0.244
# 5 Rain1       Ridge    -0.715
# 6 Temp2       Ridge     0.0426
# 7 Rain2       Ridge     1.94
# 8 Temp3       Ridge    -0.571
# 9 Rain3       Ridge     0.642
# 10 Temp4       Ridge    -0.483
# 11 (Intercept) Lm    -1637.
# 12 Year        Lm        0.876
# 13 Rain0       Lm        0.785
# 14 Temp1       Lm       -0.460
# 15 Rain1       Lm       -0.779
# 16 Temp2       Lm        0.482
# 17 Rain2       Lm        2.56
# 18 Temp3       Lm        0.0489
# 19 Rain3       Lm        0.408
# 20 Temp4       Lm       -0.656
ggplot(coefUntidyInts, aes(reorder(Coef, Beta), Beta, fill = Model)) +
  geom_bar(stat = "identity", col = "black") +
  facet_wrap(~Model) +
  theme_bw()  +
  theme(legend.position = "none")

# the intercepts are different. The intercept isn't included when
# calculating the L2 norm, but is the value of the outcome when all
# the predictors are zero. As ridge regression changes the parameter
# estimates of the predictors, the intercept changes as a result

# 3
plotHyperParsEffect(elasticTuningData, x = "s", y = "alpha",
                    z = "mse.test.mean", interpolate = "regr.kknn",
                    plot.type = "contour", show.experiments = TRUE) +
  scale_fill_gradientn(colours = terrain.colors(5)) +
  geom_point(x = tunedElasticPars$x$s, y = tunedElasticPars$x$alpha) +
  theme_bw()

plotHyperParsEffect(elasticTuningData, x = "s", y = "alpha",
                    z = "mse.test.mean", plot.type = "scatter") +
  theme_bw()

# 4
ggplot(coefUntidy, aes(reorder(Coef, Beta), Beta, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", col = "black") +
  theme_bw()

# 5
yieldOnly <- select(iowaTib, Yield)

yieldOnlyTask <- makeRegrTask(data = yieldOnly, target = "Yield")

lassoStrict <- makeLearner("regr.glmnet", lambda = 500)

loo <- makeResampleDesc("LOO")

resample("regr.lm", yieldOnlyTask, loo)
# Resampling: LOO
# Measures:             mse
# [Resample] iter 1:    272.2500000
# [Resample] iter 2:    310.9711816
# [Resample] iter 3:    52.1103516
# [Resample] iter 4:    106.3476562
# [Resample] iter 5:    775.2744141
# [Resample] iter 6:    143.1014062
# [Resample] iter 7:    957.1289062
# [Resample] iter 8:    31.0109766
# [Resample] iter 9:    14.5589941
# [Resample] iter 10:   5.1472266
# [Resample] iter 11:   5.6257910
# [Resample] iter 12:   1.0634766
# [Resample] iter 13:   104.2313379
# [Resample] iter 14:   23.4921973
# [Resample] iter 15:   4.2539062
# [Resample] iter 16:   44.9318848
# [Resample] iter 17:   47.7394629
# [Resample] iter 18:   404.3869629
# [Resample] iter 19:   117.2482910
# [Resample] iter 20:   16.1754785
# [Resample] iter 21:   3.4456641
# [Resample] iter 22:   50.6321191
# [Resample] iter 23:   158.2878516
# [Resample] iter 24:   8.9438379
# [Resample] iter 25:   16.1754785
# [Resample] iter 26:   2.7225000
# [Resample] iter 27:   8.3376562
# [Resample] iter 28:   155.7036035
# [Resample] iter 29:   272.2500000
# [Resample] iter 30:   214.4394141
# [Resample] iter 31:   185.3001563
# [Resample] iter 32:   686.1125391
# [Resample] iter 33:   718.9101562
#
#
# Aggregated Result: mse.test.mean=179.3427539
#
#
# Resample Result
# Task: yieldOnly
# Learner: regr.lm
# Aggr perf: mse.test.mean=179.3427539
# Runtime: 0.0495861

resample(lassoStrict, iowaTask, loo)
# Resampling: LOO
# Measures:             mse
# [Resample] iter 1:    272.2500000
# [Resample] iter 2:    310.9711816
# [Resample] iter 3:    52.1103516
# [Resample] iter 4:    106.3476562
# [Resample] iter 5:    775.2744141
# [Resample] iter 6:    143.1014062
# [Resample] iter 7:    957.1289062
# [Resample] iter 8:    31.0109766
# [Resample] iter 9:    14.5589941
# [Resample] iter 10:   5.1472266
# [Resample] iter 11:   5.6257910
# [Resample] iter 12:   1.0634766
# [Resample] iter 13:   104.2313379
# [Resample] iter 14:   23.4921973
# [Resample] iter 15:   4.2539062
# [Resample] iter 16:   44.9318848
# [Resample] iter 17:   47.7394629
# [Resample] iter 18:   404.3869629
# [Resample] iter 19:   117.2482910
# [Resample] iter 20:   16.1754785
# [Resample] iter 21:   3.4456641
# [Resample] iter 22:   50.6321191
# [Resample] iter 23:   158.2878516
# [Resample] iter 24:   8.9438379
# [Resample] iter 25:   16.1754785
# [Resample] iter 26:   2.7225000
# [Resample] iter 27:   8.3376563
# [Resample] iter 28:   155.7036035
# [Resample] iter 29:   272.2500000
# [Resample] iter 30:   214.4394141
# [Resample] iter 31:   185.3001563
# [Resample] iter 32:   686.1125391
# [Resample] iter 33:   718.9101562


# Aggregated Result: mse.test.mean=179.3427539
#
#
# Resample Result
# Task: iowaTib
# Learner: regr.glmnet
# Aggr perf: mse.test.mean=179.3427539
# Runtime: 0.627226

# the MSE values are identical. This is because when lambda is high
# enough, all predictors will be removed from the model, just as if
# we trained a model with no predictors

# 6
install.packages("plotmo")

library(plotmo)
# Loading required package: Formula
# Loading required package: plotrix
# Loading required package: TeachingDemos
plotres(elasticModelData)

plotres(ridgeModelData)

plotres(lassoModelData)

# the first plot shows the estimated slope for each parameter for
# different values of (log) lambda. Notice the different shape
# between ridge and LASSO
