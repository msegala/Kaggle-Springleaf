######################################################
#
# Stack a bunch of models
#
# This uses CV stacking in oppossed to Out-of-sample stacking.
# OOS stacking takes half the training data to build model 1
# and then uses model 1 on the rest of the training data and
# input to model 2.
#
# THIS IS STAGE-2 OF THE STACKING PROCEDURE
#
######################################################


### Step 1:
###  Load in the data and make neccesarry DF's and connections happen

#--------- L I B R A R Y ------------------------------------------------

library(xgboost)
library(readr)
library(Ckmeans.1d.dp)
library(caret)
library(h2o)
library(stringr)
library(pROC)

# -------- D A T A ---------------

load("~/Documents/Personal/Kaggle/Kaggle-Springleaf/SaveMe.RData")

##################################################################
#
# Add in any data from online classifiers (SGD or FTRL)
#
#################################################################

sgd_train1 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/trainSet_SGD_alpha_0.001_beta_1_L2_0.01_epoch_5.csv")
sgd_test1  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/testSet_SGD_alpha_0.001_beta_1_L2_0.01_epoch_5.csv")

sgd_train2 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/trainSet_SGD_alpha_0.001_beta_1_L2_0.1_epoch_5.csv")
sgd_test2  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/testSet_SGD_alpha_0.001_beta_1_L2_0.1_epoch_5.csv")

sgd_train3 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/trainSet_SGD_alpha_0.005_beta_10_L2_0_epoch_5.csv")
sgd_test3  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/testSet_SGD_alpha_0.005_beta_10_L2_0_epoch_5.csv")

sgd_train4 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/trainSet_SGD_alpha_0.005_beta_1_L2_0_epoch_5.csv")
sgd_test4  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/testSet_SGD_alpha_0.005_beta_1_L2_0_epoch_5.csv")

sgd_train5 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/trainSet_SGD_alpha_0.005_beta_5_L2_0_epoch_5.csv")
sgd_test5  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/testSet_SGD_alpha_0.005_beta_5_L2_0_epoch_5.csv")

sgd_train6 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/trainSet_SGD_alpha_0.01_beta_10_L2_0_epoch_5.csv")
sgd_test6  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/testSet_SGD_alpha_0.01_beta_10_L2_0_epoch_5.csv")

sgd_train7 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/trainSet_SGD_alpha_0.01_beta_1_L2_0_epoch_5.csv")
sgd_test7  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/testSet_SGD_alpha_0.01_beta_1_L2_0_epoch_5.csv")

sgd_train8 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/trainSet_SGD_alpha_0.01_beta_5_L2_0_epoch_5.csv")
sgd_test8  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/testSet_SGD_alpha_0.01_beta_5_L2_0_epoch_5.csv")

sgd_train1 <- sgd_train1[,"target"]; sgd_test1 <- sgd_test1[,"target"]; 
sgd_train2 <- sgd_train2[,"target"]; sgd_test2 <- sgd_test2[,"target"]; 
sgd_train3 <- sgd_train3[,"target"]; sgd_test3 <- sgd_test3[,"target"]; 
sgd_train4 <- sgd_train4[,"target"]; sgd_test4 <- sgd_test4[,"target"]; 
sgd_train5 <- sgd_train5[,"target"]; sgd_test5 <- sgd_test5[,"target"]; 
sgd_train6 <- sgd_train6[,"target"]; sgd_test6 <- sgd_test6[,"target"]; 
sgd_train7 <- sgd_train7[,"target"]; sgd_test7 <- sgd_test7[,"target"]; 
sgd_train8 <- sgd_train8[,"target"]; sgd_test8 <- sgd_test8[,"target"]; 


ftlr_train1 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/trainSet_FTLR_alpha_0.001_beta_1_L2_0.01_L1_0.01_epoch_5.csv")
ftlr_test1  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/testSet_FTLR_alpha_0.001_beta_1_L2_0.01_L1_0.01_epoch_5.csv")

ftlr_train2 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/trainSet_FTLR_alpha_0.005_beta_10_L2_0_L1_0_epoch_5.csv")
ftlr_test2  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/testSet_FTLR_alpha_0.005_beta_10_L2_0_L1_0_epoch_5.csv")

ftlr_train3 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/trainSet_FTLR_alpha_0.005_beta_1_L2_0_L1_0_epoch_5.csv")
ftlr_test3  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/testSet_FTLR_alpha_0.005_beta_1_L2_0_L1_0_epoch_5.csv")

ftlr_train4 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/trainSet_FTLR_alpha_0.005_beta_5_L2_0_L1_0_epoch_5.csv")
ftlr_test4  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/testSet_FTLR_alpha_0.005_beta_5_L2_0_L1_0_epoch_5.csv")

ftlr_train5 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/trainSet_FTLR_alpha_0.01_beta_10_L2_0_L1_0_epoch_5.csv")
ftlr_test5  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/testSet_FTLR_alpha_0.01_beta_10_L2_0_L1_0_epoch_5.csv")

ftlr_train6 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/trainSet_FTLR_alpha_0.01_beta_1_L2_0_L1_0_epoch_5.csv")
ftlr_test6  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/testSet_FTLR_alpha_0.01_beta_1_L2_0_L1_0_epoch_5.csv")

ftlr_train7 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/trainSet_FTLR_alpha_0.01_beta_5_L2_0_L1_0_epoch_5.csv")
ftlr_test7  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/testSet_FTLR_alpha_0.01_beta_5_L2_0_L1_0_epoch_5.csv")

ftlr_train8 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/trainSet_FTLR_alpha_0.001_beta_1_L2_0.1_L1_0.1_epoch_5.csv")
ftlr_test8  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/FTLR/testSet_FTLR_alpha_0.001_beta_1_L2_0.1_L1_0.1_epoch_5.csv")

ftlr_train1 <- ftlr_train1[,"target"]; ftlr_test1 <- ftlr_test1[,"target"]; 
ftlr_train2 <- ftlr_train2[,"target"]; ftlr_test2 <- ftlr_test2[,"target"]; 
ftlr_train3 <- ftlr_train3[,"target"]; ftlr_test3 <- ftlr_test3[,"target"]; 
ftlr_train4 <- ftlr_train4[,"target"]; ftlr_test4 <- ftlr_test4[,"target"]; 
ftlr_train5 <- ftlr_train5[,"target"]; ftlr_test5 <- ftlr_test5[,"target"]; 
ftlr_train6 <- ftlr_train6[,"target"]; ftlr_test6 <- ftlr_test6[,"target"]; 
ftlr_train7 <- ftlr_train7[,"target"]; ftlr_test7 <- ftlr_test7[,"target"]; 
ftlr_train8 <- ftlr_train8[,"target"]; ftlr_test8 <- ftlr_test8[,"target"]; 


##################################################################
#
# Add in any data from stage-1 stacking
#
##################################################################

xgboost_train1 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/Intermediate_Stacking/Intermediate_stacking_onlyXGBoost_train.csv")
xgboost_test1  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/Intermediate_Stacking/Intermediate_stacking_onlyXGBoost_test.csv")
#xgboost_train1 <- xgboost_train1[,"xgboost1"]; xgboost_test1 <- xgboost_test1[,"xgboost1"]

glm_train1 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/Intermediate_Stacking/Intermediate_stacking_onlyGLM_train.csv")
glm_test1  <-  read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/Intermediate_Stacking/Intermediate_stacking_onlyGLM_test.csv")




##################################################################
#
# We can now train the 2nd stage model with the new meta features
#
#################################################################

### Combined the orginal features with the model predictions for training set
train_meta <- cbind(train,
                    xgboost_train1,glm_train1,
                    sgd_train1,sgd_train2,sgd_train3,sgd_train4,sgd_train5,sgd_train6,sgd_train7,sgd_train8,
                    ftlr_train1,ftlr_train2,ftlr_train3,ftlr_train4,ftlr_train5,ftlr_train6,ftlr_train7,ftlr_train8)

### Combined the orginal features with the model predictions for testing set
test_meta <- cbind(test,
                   xgboost_test1,glm_test1,
                   sgd_test1,sgd_test2,sgd_test3,sgd_test4,sgd_test5,sgd_test6,sgd_test7,sgd_test8,
                   ftlr_test1,ftlr_test2,ftlr_test3,ftlr_test4,ftlr_test5,ftlr_test6,ftlr_test7,ftlr_test8)

rm(train); rm(test);


set.seed(222)
val <- sample(1:nrow(train_meta), round(0.3*nrow(train_meta))) #30% training data for validation
xgtrain = xgb.DMatrix(as.matrix(train_meta[-val,]), label = y[-val], missing = -9999)
xgval   = xgb.DMatrix(as.matrix(train_meta[val,]),  label = y[val],  missing = -9999)
gc()
watchlist <- list(eval = xgval, train = xgtrain)


param <- list(  objective           = "binary:logistic", 
                eta                 = 0.01, max_depth           = 9,  subsample   = 0.7,
                colsample_bytree    = 0.5, eval_metric         = "auc",
                min_child_weight    = 6, alpha               = 4,
                nthread=16)

clf <- xgb.train(   params    = param,     data     = xgtrain, early.stop.round    = 100,
                    nrounds   = 5000,      verbose  = 1,  print.every.n = 10, 
                    watchlist = watchlist, maximize = TRUE)


bst <- clf$bestInd
bst <- 500
### Train on full dataset
xgtrain = xgb.DMatrix(as.matrix(train_meta), label = y, missing = -9999)
watchlist <- list(train = xgtrain)

clf <- xgb.train(   params    = param,     data     = xgtrain,
                    nrounds   = bst,       verbose  = 1, print.every.n = 10, 
                    watchlist = watchlist)


xgtest <- xgb.DMatrix(as.matrix(test_meta), missing = -9999)
preds_out <- predict(clf, xgtest)

sub <- read_csv(paste0(path, "sample_submission.csv", collapse = "")) 
sub$target <- preds_out
subversion <- 1
write_csv(sub, paste0(path, "/output/Stacking/submission_Stacking_", subversion, ".csv", collapse = ""))
