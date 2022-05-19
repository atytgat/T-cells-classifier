library(dplyr)
library(plyr)
library(stringr)
library(lubridate)
library(readr)
library(caret)
require(DMwR)
require(e1071)
library(xgboost)
require(rpart)
require(randomForest)
require(bartMachine)
require(bnclassify)



# Since the classes are unbalanced, we use the balanced accuracy metrics to evaluate our models.
BCR = function(CM) {
  # Balanced accuracy metrics
  TP = CM[2,2]
  TN = CM[1,1]
  FP = CM[2,1]
  FN = CM[1,2]
  
  result = 1/2 * (TP/(TP+FN) + TN/(FP+TN))
}


# twoClassSummaryCustom function from https://stats.stackexchange.com/questions/210361/r-caret-train-rfe-optimize-for-positive-predictive-value-instead-of-accuracy-o
twoClassSummaryCustom = function (data, lev = NULL, model = NULL) 
{
  lvls <- levels(data$obs)
  if (length(lvls) > 2)
    stop(paste("Your outcome has", length(lvls), "levels. The twoClassSummary() function isn't appropriate."))
  if (!all(levels(data[, "pred"]) == lvls)) 
    stop("levels of observed and predicted data do not match")
  rocAUC <- ModelMetrics::auc(ifelse(data$obs == lev[2], 0, 
                                     1), data[, lvls[1]])
  out <- c(rocAUC,
           sensitivity(data[, "pred"], data[, "obs"], lev[1]),
           specificity(data[, "pred"], data[, "obs"], lev[2]),
           posPredValue(data[, "pred"], data[, "obs"], lev[1]),
           sensitivity(data[, "pred"], data[, "obs"], lev[1]) + specificity(data[, "pred"], data[, "obs"], lev[2]))
  names(out) <- c("ROC", "Sens", "Spec", "Prec", "BCR")
  out
}


### Load and prepare the dataset
load("C:/Users/alext/Documents/full_df.processed.RData")

# preprocess : na -> 0
full_df = read.csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M1/Q2/LINGI2262 - Machine Learning Classification and Evaluation/Inginious Sessions/Session 5/A5-2020_datasets-20200429/ML-A5-2020_train.csv", row.names = 1)
test.df = read.csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M1/Q2/LINGI2262 - Machine Learning Classification and Evaluation/Inginious Sessions/Session 5/A5-2020_datasets-20200429/ML-A5-2020_test.csv", row.names = 1)

genes.test = test.df[,1:23370]
genes.test[is.na(genes.test)] <- 0
test_df.processed = cbind(genes.test, test.df[,23371:23384])

genes.full = full_df[,1:23370]
genes.full[is.na(genes.full)] <- 0
full_df.processed = cbind(genes.full, full_df[,23371:23385])



set.seed(998)

# Model selection is performed using the k-fold method with 5 folds.
kfolds = createFolds(full_df.processed$label, k = 5)



#--------------------- 1.TRAIN-VALID SETS CREATION ---------------------------#


train_index = kfolds$Fold1
train.df <- full_df.processed[-train_index,]
valid.df  <- full_df.processed[train_index,]
test.df <- test_df.processed


#------------------------ 2.FEATURE SELECTION --------------------------------#


# keep only the genes features
genes.train = train.df[,1:23370]
genes.valid = valid.df[,1:23370]
genes.test = test.df[,1:23370]

# supression of columns with only 0 in training set (necessary for PCA)
# from https://stackoverflow.com/questions/21530168/remove-columns-with-zero-values-from-a-dataframe
genes.valid = genes.valid[,-(which(colSums(genes.train)==0))]
genes.test = genes.test[,-(which(colSums(genes.train)==0))]
genes.train = genes.train[,-(which(colSums(genes.train)==0))]


# PCA
genes.pr <- prcomp(x = genes.train, center = TRUE, scale = TRUE, rank = 50)
train.pr = as.data.frame(predict(genes.pr, genes.train))
valid.pr = as.data.frame(predict(genes.pr, genes.valid))
test.pr = as.data.frame(predict(genes.pr, genes.test))

preProcValues = preProcess(train.pr, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train.pr)
validTransformed <- predict(preProcValues, valid.pr)
testTransformed <- predict(preProcValues, test.pr)


# new train-valid sets with PC and additional features as FACTOR
training.set = cbind(trainTransformed, train.df[,23371:23385])
validation.set = cbind(validTransformed, valid.df[,23371:23385])
test.set = cbind(testTransformed, test.df[,23371:23384])

# factor to numeric https://stackoverflow.com/questions/27528907/how-to-convert-data-frame-column-from-factor-to-numeric/27528953
cols = c('patient', 'tissue', 'level.mito', 'level.ribo', 'low.yield', 'marker.A', 'marker.B', 'marker.C', 'marker.D', 'marker.E', 'marker.F', 'marker.G', 'label')
training.set[cols] <- lapply(training.set[cols], factor)
validation.set[cols] <- lapply(validation.set[cols], factor)  

training.set[cols] = lapply(training.set[cols], function(x) as.factor(as.numeric(x)))
validation.set[cols] = lapply(validation.set[cols], function(x) as.factor(as.numeric(x)))

levels(training.set$label) <- c("first_class", "second_class")
levels(validation.set$label) <- c("first_class", "second_class")

cols = c('patient', 'tissue', 'level.mito', 'level.ribo', 'low.yield', 'marker.A', 'marker.B', 'marker.C', 'marker.D', 'marker.E', 'marker.F', 'marker.G')
test.set[cols] = lapply(test.set[cols], factor)
test.set[cols] = lapply(test.set[cols], function(x) as.factor(as.numeric(x)))


#------------------------- 3. BALANCING CLASSES --------------------------------#


table(training.set$label)
set.seed(128)
training.smote <- SMOTE(label~., training.set, perc.over = 300, perc.under = 200, k = 7)
table(training.smote$label)


#------------------------- 4.1 GBM FITTING --------------------------------------#
# GBM tuning from
# https://topepo.github.io/caret/model-training-and-tuning.html

training.set.final = training.smote

x.train = subset(training.set.final, select = -label)
y.train = training.set.final$label
x.valid = subset(validation.set, select = -label)
y.valid = validation.set$label


gbm.control = trainControl(
  method = 'boot632',
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummaryCustom
)

### First model

set.seed(123)
gbmfit_1 <- train(label ~ .,
             data = training.set.final,
             method = "gbm",
             metric = "BCR",
             trControl = gbm.control,
             verbose = FALSE, 
             preProc = c("center", "scale"),
             tuneGrid =  expand.grid(interaction.depth = c(3, 5, 7), 
                                         n.trees = 100, 
                                         shrinkage = c(0.1,0.01),
                                         n.minobsinnode = c(20,40,60)
             
))

### Prediction
valid_predict = predict(gbmfit_1, x.valid)
BCR.valid = table(valid_predict, y.valid)

confusionMatrix(data = valid_predict, y.valid)
gbmfit_1$bestTune

# best BCR = 0.7024
# n.trees interaction.depth shrinkage n.minobsinnode
# 16     100                 7       0.1             20


### Second model

set.seed(123)
gbmfit_2 <- train(label ~ .,
                  data = training.set.final,
                  method = "gbm",
                  metric = "BCR",
                  trControl = gbm.control,
                  verbose = FALSE, 
                  preProc = c("center", "scale"),
                  tuneGrid =  expand.grid(interaction.depth = c(5, 7, 9, 11), 
                                          n.trees = c(50,100), 
                                          shrinkage = c(0.1,0.05),
                                          n.minobsinnode = c(20,40,60)
                                          
))

### Prediction
valid_predict = predict(gbmfit_2, x.valid)
BCR.valid = table(valid_predict, y.valid)

confusionMatrix(data = valid_predict, y.valid)
gbmfit_2$bestTune
# 0.7079
# n.trees interaction.depth shrinkage n.minobsinnode
#37     100                 7       0.1             20



#------------------------- 4.2 XGBTREE FITTING --------------------------------------#
# XGBTree tuning from https://github.com/topepo/caret/blob/master/RegressionTests/Code/xgbTree.R

training.set.final = training.smote

x.train = subset(training.set.final, select = -label)
y.train = training.set.final$label
x.valid = subset(validation.set, select = -label)
y.valid = validation.set$label


xgb.control = trainControl(
  method = 'boot632',
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummaryCustom
)


### First model


set.seed(123)
xbgfit_1 <- train(label ~ .,
             data = training.set.final,
             method = "xgbTree",
             metric = "BCR",
             trControl = xgb.control,
             verbose = FALSE, 
             preProc = c("center", "scale"),
             tuneGrid =expand.grid(nrounds=100,
                                   eta=c(0.01, 0.05, 0.1),
                                   max_depth=c(5, 7, 9),
                                   gamma= 1,
                                   colsample_bytree=c(0.4, 0.7, 1.0),
                                   subsample=c(0.5, 0.7),
                                   min_child_weight=c(0.5, 1, 1.5)
            
))


### Prediction
valid_predict = predict(xbgfit_1, x.valid)
BCR.valid = table(valid_predict, y.valid)

confusionMatrix(data = valid_predict, y.valid)
xbgfit_1$bestTune
# 0.7688
#nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
#116     100         5 0.1     1              0.7              0.5       0.7


#------------------------- 4.3 RANGER FITTING --------------------------------------#
# Ranger tuning from https://compgenomr.github.io/book/trees-and-forests-random-forests-in-action.html#decision-trees

training.set.final = training.smote

x.train = subset(training.set.final, select = -label)
y.train = training.set.final$label
x.valid = subset(validation.set, select = -label)
y.valid = validation.set$label


ranger.control = trainControl(
  method = 'boot632',
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummaryCustom
)


### First model
rangerGrid_1 <-  
set.seed(123)
rangerfit_1 <- train(label ~ .,
             data = training.set.final,
             method = "ranger",
             metric = "BCR",
             trControl = ranger.control,
             preProc = c("center", "scale"),
             tuneGrid = expand.grid(mtry=c(10,20,30,40,50,60),
                                    min.node.size = c(3,5,7,9),
                                    splitrule="gini"
))


### Prediction
valid_predict = predict(rangerfit_1, x.valid)
BCR.valid = table(valid_predict, y.valid)

confusionMatrix(data = valid_predict, y.valid)
rangerfit_1$bestTune
# 0.7480 
#  mtry splitrule min.node.size
# 2   10      gini             5



#------------------------- 4.4 DART FITTING --------------------------------------#
# Dart tuning from https://github.com/topepo/caret/blob/master/RegressionTests/Code/xgbDART.R

training.set.final = training.smote

x.train = subset(training.set.final, select = -label)
y.train = training.set.final$label
x.valid = subset(validation.set, select = -label)
y.valid = validation.set$label

dart.control = trainControl(
  method = 'boot632',
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummaryCustom
)


### First model

set.seed(123)
dartfit_1 <- train(label ~ .,
                   data = training.set.final,
                   method = "xgbDART",
                   metric = "BCR",
                   trControl = dart.control,
                   preProc = c("center", "scale"),
                   tuneGrid = expand.grid(nrounds = 100, 
                                          max_depth = 5,
                                          eta = c(0.1,0.3),
                                          rate_drop = 0.10,
                                          skip_drop = 0.10,
                                          colsample_bytree = 0.90,
                                          min_child_weight = c(2,5),
                                          subsample = c(0.5,0.75),
                                          gamma = 0.10))

### Prediction
valid_predict = predict(dartfit_1, x.valid)
BCR.valid = table(valid_predict, y.valid)

confusionMatrix(data = valid_predict, y.valid)
dartfit_1$bestTune
#  0.7869 
#nrounds max_depth eta gamma subsample colsample_bytree rate_drop skip_drop min_child_weight
#1     100         5 0.3   0.1      0.75              0.9       0.1       0.1                2


### Second model

set.seed(123)
dartfit_2 <- train(label ~ .,
                   data = training.set.final,
                   method = "xgbDART",
                   metric = "BCR",
                   trControl = dart.control,
                   preProc = c("center", "scale"),
                   tuneGrid = expand.grid(nrounds = 100, 
                                          max_depth = 5,
                                          eta = c(0.1,0.3),
                                          rate_drop = 0.10,
                                          skip_drop = 0.10,
                                          colsample_bytree = 0.90,
                                          min_child_weight = c(2,5),
                                          subsample = c(0.5,0.75),
                                          gamma = 0.10))

### Prediction
valid_predict = predict(dartfit_2, x.valid)
BCR.valid = table(valid_predict, y.valid)

confusionMatrix(data = valid_predict, y.valid)
dartfit_2$bestTune
#0.7397     
#nrounds max_depth eta gamma subsample colsample_bytree rate_drop skip_drop min_child_weight
#6     100         5 0.3   0.1      0.75              0.9       0.1       0.1                2


#------------------------- 4.5 LINXGB FITTING --------------------------------------#


training.set.final = training.smote

x.train = subset(training.set.final, select = -label)
y.train = training.set.final$label
x.valid = subset(validation.set, select = -label)
y.valid = validation.set$label

linxgb.control = trainControl(
  method = 'boot632',
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummaryCustom
)


### First model

set.seed(123)
linxgbfit_1 <- train(label ~ .,
                   data = training.set.final,
                   method = "xgbLinear",
                   metric = "BCR",
                   trControl = linxgb.control,
                   preProc = c("center", "scale"),
                   tuneGrid = expand.grid(nrounds = 100, 
                                          lambda = c(0.1,0.5,1,1.5), 
                                          alpha = c(0.01,0.05,0.1,0.3),
                                          eta = c(0.01,0.05,0.1,0.3)
))

### Prediction
valid_predict = predict(linxgbfit_1, x.valid)
BCR.valid = table(valid_predict, y.valid)

confusionMatrix(data = valid_predict, y.valid)
linxgbfit_1$bestTune
# 0.7523
#nrounds lambda alpha  eta
#49     100    1.5  0.01 0.01

### Second model
set.seed(123)
linxgbfit_2 <- train(label ~ .,
                     data = training.set.final,
                     method = "xgbLinear",
                     metric = "BCR",
                     trControl = linxgb.control,
                     preProc = c("center", "scale"),
                     tuneGrid = expand.grid(nrounds = 100, 
                                            lambda = c(0.1,0.5), 
                                            alpha = c(0.01,0.1,0.3),
                                            eta = c(0.1,0.3,0.5)
))


### Prediction
valid_predict = predict(linxgbfit_2, x.valid)
BCR.valid = table(valid_predict, y.valid)

confusionMatrix(data = valid_predict, y.valid)
linxgbfit_2$bestTune
# 0.7315
# nrounds lambda alpha eta
# 13     100    0.5   0.1 0.1

#------------------------- 4.6 ADA FITTING --------------------------------------#


training.set.final = training.smote

x.train = subset(training.set.final, select = -label)
y.train = training.set.final$label
x.valid = subset(validation.set, select = -label)
y.valid = validation.set$label

ada.control = trainControl(
  method = 'boot632',
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummaryCustom
)


### First model
set.seed(123)
adafit_1 <- train(x.train, y.train,
                  method = "ada",
                  metric = "BCR",
                  trControl = ada.control,
                  preProc = c("center", "scale"),
                  tuneGrid = expand.grid(iter = 100, 
                              maxdepth = c(3,5,7), 
                              nu = c(0.1,0.3)
))

### Prediction
valid_predict = predict(adafit_1, x.valid)
BCR.valid = table(valid_predict, y.valid)

confusionMatrix(data = valid_predict, y.valid)
adafit_1$bestTune
# 0.7605
#iter maxdepth  nu
#5  100        5 0.3


### Second model

set.seed(123)
adafit_2 <- train(x.train, y.train,
                     method = "ada",
                     metric = "BCR",
                     trControl = ada.control,
                     preProc = c("center", "scale"),
                     tuneGrid = expand.grid(iter = 150, 
                                            maxdepth = 5, 
                                            nu = 0.1)
)

### Prediction
valid_predict = predict(adafit_2, x.valid)
BCR.valid = table(valid_predict, y.valid)

confusionMatrix(data = valid_predict, y.valid)
adafit_2$bestTune
# 0.7217
#iter maxdepth  nu
#1  150        5 0.1



#------------------------- 5. ENSEMBLE LEARNING --------------------------------------#
# code from https://compgenomr.github.io/book/other-supervised-algorithms.html#ensemble-learning



gbmPred=as.character(predict(gbmfit_2,x.valid)) # 0.7079

xbgPred=as.character(predict(xbgfit_1,x.valid)) # 0.7688

rangerPred=as.character(predict(rangerfit_1,x.valid)) # 0.7480

dartPred=as.character(predict(dartfit_1,x.valid)) # 0.7869 

linxgbPred=as.character(predict(linxgbfit_1,x.valid)) # 0.7523

adaPred=as.character(predict(adafit_1,x.valid)) # 0.7605


# do voting for class labels
# code finds the most frequent class label per row
# all
votingPred_all=apply(cbind(gbmPred,xbgPred,rangerPred,dartPred,linxgbPred,adaPred),1,function(x) names(which.max(table(x))))

# only the best
votingPred_best=apply(cbind(xbgPred,linxgbPred,dartPred,adaPred),1,function(x) names(which.max(table(x))))

# check accuracy
confusionMatrix(data= as.factor(votingPred_best), reference=y.valid)
# best BCR =0.7826  


#-------------------------- 6. TEST PREDICTION ---------------------------------------#




x.test = test.set



gbmPred=as.character(predict(gbmfit_2,test.set))

xbgPred=as.character(predict(xbgfit_1,test.set))

rangerPred=as.character(predict(rangerfit_1,test.set))

dartPred=as.character(predict(dartfit_1,test.set))

linxgbPred=as.character(predict(linxgbfit_1,test.set))

adaPred=as.character(predict(adafit_1,test.set))

# all
votingPred_all=apply(cbind(gbmPred,xbgPred,rangerPred,dartPred,linxgbPred,adaPred),1,function(x) names(which.max(table(x))))

# only the best
votingPred_best=apply(cbind(xbgPred,linxgbPred,dartPred,adaPred),1,function(x) names(which.max(table(x))))


test_predict = as.factor(votingPred_all)

Prediction = revalue(test_predict, c("first_class"="-1", "second_class"="1"))
Prediction

final.df = cbind(test.df, Prediction)
table(final.df$Prediction)

write.csv(final.df['Prediction'], "prediction_final_best.csv")
