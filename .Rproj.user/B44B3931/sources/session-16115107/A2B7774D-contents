###############################################
#          SOURCE CODE FOR CHAPTER 9          #
###############################################

# LOAD PACKAGES ----
library(mlr)

library(tidyverse)

# LOAD DATA ----
data(Ozone, package = "mlbench")

ozoneTib <- as_tibble(Ozone)

names(ozoneTib) <- c("Month", "Date", "Day", "Ozone", "Press_height",
                     "Wind", "Humid", "Temp_Sand", "Temp_Monte",
                     "Inv_height", "Press_grad", "Inv_temp", "Visib")

ozoneTib
# A tibble: 366 × 13
# Month Date  Day   Ozone Press_height  Wind Humid Temp_Sand Temp_Monte Inv_height Press_grad Inv_temp
# <fct> <fct> <fct> <dbl>        <dbl> <dbl> <dbl>     <dbl>      <dbl>      <dbl>      <dbl>    <dbl>
# 1 1     1     4         3         5480     8    20        NA       NA         5000        -15     30.6
# 2 1     2     5         3         5660     6    NA        38       NA           NA        -14     NA
# 3 1     3     6         3         5710     4    28        40       NA         2693        -25     47.7
# 4 1     4     7         5         5700     3    37        45       NA          590        -24     55.0
# 5 1     5     1         5         5760     3    51        54       45.3       1450         25     57.0
# 6 1     6     2         6         5720     4    69        35       49.6       1568         15     53.8
# 7 1     7     3         4         5790     6    19        45       46.4       2631        -33     54.1
# 8 1     8     4         4         5790     3    25        55       52.7        554        -28     64.8
# 9 1     9     5         6         5700     3    73        41       48.0       2083         23     52.5
# 10 1     10    6         7         5700     3    59        44       NA         2654         -2     48.4
# ℹ 356 more rows
# ℹ 1 more variable: Visib <dbl>
# ℹ Use `print(n = ...)` to see more rows
ozoneClean <- mutate_all(ozoneTib, as.numeric) %>%
  filter(is.na(Ozone) == FALSE)

ozoneClean
# A tibble: 361 × 13
# Month  Date   Day Ozone Press_height  Wind Humid Temp_Sand Temp_Monte Inv_height Press_grad Inv_temp
# <dbl> <dbl> <dbl> <dbl>        <dbl> <dbl> <dbl>     <dbl>      <dbl>      <dbl>      <dbl>    <dbl>
# 1     1     1     4     3         5480     8    20        NA       NA         5000        -15     30.6
# 2     1     2     5     3         5660     6    NA        38       NA           NA        -14     NA
# 3     1     3     6     3         5710     4    28        40       NA         2693        -25     47.7
# 4     1     4     7     5         5700     3    37        45       NA          590        -24     55.0
# 5     1     5     1     5         5760     3    51        54       45.3       1450         25     57.0
# 6     1     6     2     6         5720     4    69        35       49.6       1568         15     53.8
# 7     1     7     3     4         5790     6    19        45       46.4       2631        -33     54.1
# 8     1     8     4     4         5790     3    25        55       52.7        554        -28     64.8
# 9     1     9     5     6         5700     3    73        41       48.0       2083         23     52.5
# 10     1    10     6     7         5700     3    59        44       NA         2654         -2     48.4
# ℹ 351 more rows
# ℹ 1 more variable: Visib <dbl>
# ℹ Use `print(n = ...)` to see more rows
# PLOT DATA ----
ozoneUntidy <- gather(ozoneClean, key = "Variable", value = "Value", -Ozone)

ozoneUntidy
# A tibble: 4,332 × 3
# Ozone Variable Value
# <dbl> <chr>    <dbl>
# 1     3 Month        1
# 2     3 Month        1
# 3     3 Month        1
# 4     5 Month        1
# 5     5 Month        1
# 6     6 Month        1
# 7     4 Month        1
# 8     4 Month        1
# 9     6 Month        1
# 10     7 Month        1
# ℹ 4,322 more rows
# ℹ Use `print(n = ...)` to see more rows
ggplot(ozoneUntidy, aes(Value, Ozone)) +
  facet_wrap(~ Variable, scale = "free_x") +
  geom_point() +
  geom_smooth() +
  geom_smooth(method = "lm", col = "red") +
  theme_bw()

ggplot(ozoneUntidy, aes(Value, Ozone)) +
  facet_wrap(~ Variable, scale = "free_x") +
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth() +
  geom_smooth(method = "lm", col = "red") +
  theme_bw()

#ggsave("Ozone plots.pdf", width = 10, height = 6)

# IMPUTE MISSING VALUES ----
?imputations

imputeMethod <- imputeLearner("regr.rpart")

ozoneImp <- impute(as.data.frame(ozoneClean),
                   classes = list(numeric = imputeMethod))

ozoneImp
# MAKE TASK AND LEARNER ----
ozoneTask <- makeRegrTask(data = ozoneImp$data, target = "Ozone")

lin <- makeLearner("regr.lm")

# FEATURE SELECTION FILTER METHOD ----
#install.packages("FSelector")
listFilterMethods()
filterVals <- generateFilterValuesData(ozoneTask,
                                       method = "linear.correlation")

filterVals$data
# name    type             filter      value
# 1:        Month numeric linear.correlation 0.05371420
# 2:         Date numeric linear.correlation 0.08205094
# 3:          Day numeric linear.correlation 0.04151444
# 4: Press_height numeric linear.correlation 0.58752354
# 5:         Wind numeric linear.correlation 0.00468138
# 6:        Humid numeric linear.correlation 0.45148094
# 7:    Temp_Sand numeric linear.correlation 0.76977674
# 8:   Temp_Monte numeric linear.correlation 0.74159034
# 9:   Inv_height numeric linear.correlation 0.57563391
# 10:   Press_grad numeric linear.correlation 0.23331823
# 11:     Inv_temp numeric linear.correlation 0.72712691
# 12:        Visib numeric linear.correlation 0.41471463
plotFilterValues(filterVals) + theme_bw() + theme(axis.text.x = element_text(angle = 90))
#ggsave("Filtervals.pdf", width = 10, height = 6)

#ozoneFiltTask <- filterFeatures(ozoneTask, fval = filterVals, abs = 6) #6 most important features
#ozoneFiltTask <- filterFeatures(ozoneTask, fval = filterVals, per = 0.25) #25% most important features
#ozoneFiltTask <- filterFeatures(ozoneTask, fval = filterVals, threshold = 0.2) #variables with contribution > 2

# FILTER WRAPPER ----
filterWrapper <- makeFilterWrapper(learner = lin,
                                   fw.method = "linear.correlation")

lmParamSpace <- makeParamSet(
  makeIntegerParam("fw.abs", lower = 1, upper = 12)
)

gridSearch <- makeTuneControlGrid()

kFold <- makeResampleDesc("CV", iters = 10)

tunedFeats <- tuneParams(filterWrapper,
                         task = ozoneTask,
                         resampling = kFold,
                         par.set = lmParamSpace,
                         control = gridSearch)

tunedFeats

# MAKE NEW TASK AND TRAIN MODEL FOR FILTER METHOD ----
filteredTask <- filterFeatures(ozoneTask, fval = filterVals,
                               abs = unlist(tunedFeats$x))

filteredModel <- train(lin, filteredTask)

# FEATURE SELECTION WRAPPER METHOD ----
featSelControl <- makeFeatSelControlSequential(method = "sfbs")

selFeats <- selectFeatures(learner = lin,
                           task = ozoneTask,
                           resampling = kFold,
                           control = featSelControl)

selFeats

# MAKE NEW TASK AND TRAIN MODEL FOR THE WRAPPER METHOD ----
ozoneSelFeat <- ozoneImp$data[, c("Ozone", selFeats$x)]

ozoneSelFeatTask <- makeRegrTask(data = ozoneSelFeat, target = "Ozone")

wrapperModel <- train(lin, ozoneSelFeatTask)

# MAKE IMPUTATION WRAPPER FOR CROSS-VALIDATION ----
imputeMethod <- imputeLearner("regr.rpart")

imputeWrapper <- makeImputeWrapper(lin,
                                   classes = list(numeric = imputeMethod))

# MAKE FEATURE SELECTION WRAPPER FOR CROSS-VALIDATION ----
featSelControl <- makeFeatSelControlSequential(method = "sfbs")

featSelWrapper <- makeFeatSelWrapper(learner = imputeWrapper,
                                     resampling = kFold,
                                     control = featSelControl)

# CROSS-VALIDATE MODEL BUILDING PROCESS ----
library(parallel)
library(parallelMap)

ozoneTaskWithNAs <- makeRegrTask(data = ozoneClean, target = "Ozone")

kFold3 <- makeResampleDesc("CV", iters = 3)

parallelStartSocket(cpus = detectCores())

lmCV <- resample(featSelWrapper, ozoneTaskWithNAs, resampling = kFold3) #~ 1.5 min

parallelStop()

lmCV


# LOOK AT MODEL DIAGNOSTICS ----
wrapperModelData <- getLearnerModel(wrapperModel)
summary(wrapperModelData)

par(mfrow = c(2, 2))
plot(wrapperModelData)
par(mfrow = c(1, 1))

# SOLUTIONS TO EXERCISES ----
# 1
filterValsForest <- generateFilterValuesData(ozoneTask,
                                             method = "randomForestSRC_importance")

filterValsForest$data

plotFilterValues(filterValsForest) + theme_bw()

# 2
filterWrapperDefault <- makeFilterWrapper(learner = lin)

tunedFeats <- tuneParams(filterWrapperDefault, task = ozoneTask,
                         resampling = kFold, par.set = lmParamSpace,
                         control = gridSearch)

tunedFeats

# the defaut filter statistic (randomForestSRC) tends to select fewer
# predictors in this case, but the linear.correlation statistic was faster

# 3
filterWrapperImp <- makeFilterWrapper(learner = imputeWrapper,
                                   fw.method = "linear.correlation")
filterParam <- makeParamSet(
  makeIntegerParam("fw.abs", lower = 1, upper = 12)
)

tuneWrapper <- makeTuneWrapper(learner = filterWrapperImp,
                               resampling = kFold,
                               par.set = filterParam,
                               control = gridSearch)

filterCV <- resample(tuneWrapper, ozoneTask, resampling = kFold)

filterCV

# we have a similar MSE estimate for the filter method
# but it is considerably faster than the wrapper method. No free lunch!

