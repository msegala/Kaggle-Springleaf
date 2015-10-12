######################################################
#
# Stack a bunch of models
#
# This uses CV stacking in oppossed to Out-of-sample stacking.
# OOS stacking takes half the training data to build model 1
# and then uses model 1 on the rest of the training data and
# input to model 2.
#
###########################


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

#load("~/Documents/Personal/Kaggle/Kaggle-Springleaf/SaveMe.RData")

cat("reading the train and test data\n")
path = "/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/"

train <- read_csv(paste0(path, "train.csv", collapse = ""))
y <- train$target
ID <- train$ID
train <- train[,-c(1, length(train))]

test <- read_csv(paste0(path, "test.csv", collapse = ""))
test_ID <- test$ID
test <- test[,-1]


##### MAKE IT SMALL FOR TESTING.....
train = train[1:10000,]
y = y[1:10000]
test = test[1:10000,]



#
#### These are Date columns, turn them into floats.....
#
datecolumns = c("VAR_0073", "VAR_0075", "VAR_0156", "VAR_0157", "VAR_0158", "VAR_0159", "VAR_0166", 
                "VAR_0167", "VAR_0168", "VAR_0176", "VAR_0177", "VAR_0178", "VAR_0179", "VAR_0204", "VAR_0217")
train_cropped <- train[datecolumns]
train_cc <- data.frame(apply(train_cropped, 2, function(x) as.double(strptime(x, format='%d%b%y:%H:%M:%S', tz="UTC")))) #2 = columnwise
for (dc in datecolumns){
  train[dc] <- NULL
  train[dc] <- train_cc[dc]
}
train_cc <- NULL;train_cropped <- NULL;gc()

test_cropped <- test[datecolumns]
test_cc <- data.frame(apply(test_cropped, 2, function(x) as.double(strptime(x, format='%d%b%y:%H:%M:%S', tz="UTC")))) #2 = columnwise
for (dc in datecolumns){
  test[dc] <- NULL
  test[dc] <- test_cc[dc]
}
test_cc <- NULL;test_cropped <- NULL;gc()
rm(test_cc); rm(dc); rm(test_cropped); rm(train_cc); rm(train_cropped);

#
#### Replacing categorical features with factors
#
cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in 1:ncol(train)) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))    
  }
}

#
#### Check for constant columns and remove them
#
col_ct = sapply(train, function(x) length(unique(x)))
cat("Constant feature count:", length(col_ct[col_ct==1]))
train = train[, !names(train) %in% names(col_ct[col_ct==1])]
test  = test[,  !names(test)  %in% names(col_ct[col_ct==1])]  


cat("replacing missing values with -9999\n")
train[is.na(train)] <- -9999
test[is.na(test)]   <- -9999

train[train<0] <- -9999
test[test<0]   <- -9999







set.seed(222)
val <- sample(1:nrow(train), 1000) #10% training data for validation


# -------- h2o -------------------

h2o.shutdown(h2oServer)
## Launch H2O directly on localhost, go to http://localhost:54321/ to check Jobs/Data/Models etc.!
h2oServer <- h2o.init(nthreads = -1)

## Attach the labels to the training data
#trainWL <- cbind(train, y)
#trainWL <- as.h2o(h2oServer, trainWL)
#trainWL <- h2o.assign(trainWL, "trainWL")
testWL <- as.h2o(h2oServer, test)
testWL <- h2o.assign(testWL, "testWL")


# -------- XGBoost -------------------

#xgtrain = xgb.DMatrix(as.matrix(train), label = y, missing = -9999)
#gc()


# -------- Parameters ----------------
n_folds = 4
verbose = TRUE
shuffle = FALSE


# -------- Setup Stacking ------------

### Returns train inidices for n_folds using StratifiedKFold
skf = createFolds(y, k = n_folds , list = TRUE, returnTrain = TRUE)

### Create a list of models to run
clfs  <- c("xgboost1", "xgboost2", "rf1", "gbm1", "glm1", "glm2", "glm3")
types <- c("xgboost",  "xgboost",  "h2o", "h2o",  "h2o",  "h2o",   "h2o")

### Pre-allocate the data
### For each model, add a column with N rows for each model
dataset_blend_train = matrix(0, nrow(train), length(clfs))
dataset_blend_test  = matrix(0, nrow(test), length(clfs))


### Loop over the models to perform stage 1 stacking
j <- 0 
for (clf in clfs){
  j <- j + 1
  tmp_type <- types[j] # What kind of model are we running...
  cat(paste("Model:",j,"Type:",tmp_type,"Name:",clf,"\n"))
  
  ### Create a tempory array that is (Holdout_Size, N_Folds).
  ### Number of testing data x Number of folds , we will take the mean of the predictions later
  dataset_blend_test_j = matrix(0, nrow(test), length(skf))
  #cat(paste(nrow(dataset_blend_test_j),ncol(dataset_blend_test_j)))
  
  ### Loop over the folds
  i <- 0
  for (sk in skf){
    i <- i + 1
    cat(paste("Fold", i,"\n"))
    
    ### Extract and fit the train/test section for each fold    
    tmp_train <- unlist(skf[i])
    X_train = train[tmp_train,]
    y_train = y[tmp_train]
    X_test  = train[-tmp_train,]
    y_test  = y[-tmp_train]
    
    ### Stupid hack to fit the model
    if (tmp_type == "xgboost"){

      if(i==1){cat("Running an XGBoost model....\n")}
      xgtrain_tmp = xgb.DMatrix(as.matrix(X_train), label = y_train, missing = -9999)
      xgval_tmp   = xgb.DMatrix(as.matrix(X_test),  label = y_test,  missing = -9999)
      gc()
      watchlist <- list(eval = xgval_tmp, train = xgtrain_tmp)
      
      if (clf == "xgboost1"){
        if(i==1){cat("Running xgboost1....\n")}
        param <- list(  objective = "binary:logistic", eta = 0.01, max_depth = 8,  
                        subsample = 0.72, colsample_bytree = 0.74, eval_metric = "auc")
        
        mod <- xgb.train(   params = param, data = xgtrain_tmp, nrounds = 2, verbose = 1, 
                            early.stop.round = 18, watchlist = watchlist, maximize = TRUE)
      }
      else if (clf == "xgboost2"){
        if(i==1){cat("Running xgboost2....\n")}
        param <- list(  objective = "binary:logistic", eta = 0.1, max_depth = 18,  
                        subsample = 0.72, colsample_bytree = 0.74, eval_metric = "auc")
        
        mod <- xgb.train(   params = param, data = xgtrain_tmp, nrounds = 2, verbose = 2, 
                            early.stop.round = 18, watchlist = watchlist, maximize = TRUE)
      }
      
      rm(xgtrain_tmp)
      rm(xgval_tmp)
                  
      ### Predict the probability of current folds test set and store results.
      train_pred <- predict(mod, xgb.DMatrix(as.matrix(X_test), missing = -9999), ntreelimit = mod$bestInd)
      dataset_blend_train[-tmp_train, j] <- train_pred
      cat("Local CV Score:", auc(y_test,train_pred),"\n")
      
      ### Predict the probabilty for the true test set and store results
      dataset_blend_test_j[, i] <- predict(mod, xgb.DMatrix(as.matrix(test), missing = -9999), ntreelimit = mod$bestInd)
    }
    else if (tmp_type == "h2o"){
    
      if(i==1){cat("Running an H2o model....\n")}
      X_trainWL <- cbind(X_train, y_train)
      X_trainWL <- as.h2o(h2oServer, X_trainWL)
      X_trainWL <- h2o.assign(X_trainWL, 'X_trainWL')
      X_testWL  <- as.h2o(h2oServer, X_test)
      X_testWL  <- h2o.assign(X_testWL,  'X_testWL')
      
      if (clf == "rf1"){
        if(i==1){cat("Running rf1....\n")}
        mod <-  h2o.randomForest(training_frame=X_trainWL,  
                                 key = "rf1", x=c(1:(ncol(X_trainWL)-1)), y=ncol(X_trainWL),
                                 type="BigData", ntrees = 2, max_depth = 8, seed=42)
      }
      else if (clf == "gbm1"){
        if(i==1){cat("Running gbm1....\n")}
        mod <-  h2o.gbm(training_frame=X_trainWL,  
                        key = "gbm1", x=c(1:(ncol(X_trainWL)-1)), y=ncol(X_trainWL),
                        distribution = "AUTO", type="BigData", ntrees = 2, max_depth = 10, learn_rate = 0.01)
      }
      else if (clf == "glm1"){
        if(i==1){cat("Running glm1....\n")}
        mod <-  h2o.glm(training_frame=X_trainWL,  
                        x=c(1:(ncol(X_trainWL)-1)), y=ncol(X_trainWL),
                        family="gaussian", alpha = 0, lambda = 1e-07)
      }
      else if (clf == "glm2"){
        if(i==1){cat("Running glm2....\n")}
        mod <-  h2o.glm(training_frame=X_trainWL,  
                        x=c(1:(ncol(X_trainWL)-1)), y=ncol(X_trainWL),
                        family="gaussian", alpha = 1, lambda = 1e-07)
      }
      else if (clf == "glm3"){
        if(i==1){cat("Running glm3....\n")}
        mod <-  h2o.glm(training_frame=X_trainWL,  
                        x=c(1:(ncol(X_trainWL)-1)), y=ncol(X_trainWL),
                        family="gaussian", alpha = 0.5, lambda = 1e-07)
      }
      
      
      ### Predict the probability of current folds test set and store results.
      train_pred <- as.data.frame(h2o.predict(mod, X_testWL))[,1]
      dataset_blend_train[-tmp_train, j] <- train_pred
      cat("Local CV Score:", auc(y_test,train_pred),"\n")
                    
      ### Predict the probabilty for the true test set and store results
      dataset_blend_test_j[, i] <- as.data.frame(h2o.predict(mod, testWL))[,1]      
    }
    
    ### Predict the probability of current folds test set and store results.
    ### This output will be the basis for our blended classifier to train against,
    ### which is also the output of our classifiers
    #dataset_blend_train[-tmp_train, j] <- predict(mod, X_test, n.trees=best.iter, type="response")
    
    ### Predict the probabilty for the true test set and store results
    #dataset_blend_test_j[, i] <- predict(mod, test, n.trees=best.iter, type="response")
  }
  
  ### Take mean of final holdout set folds
  dataset_blend_test[,j] = rowMeans(dataset_blend_test_j)
  cat("\n")
}


##################################################################
#
# Add in any data from online classifiers (SGD or FTRL)
#
#################################################################

sgd_train1 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/trainSet_SGD_alpha_0.005_beta_1_L2_0_epoch_5.csv")
sgd_test1 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/testSet_SGD_alpha_0.005_beta_1_L2_0_epoch_5.csv")

sgd_train2 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/trainSet_SGD_alpha_0.005_beta_5_L2_0_epoch_5.csv")
sgd_test2 <- read.csv("/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/output/SGD/testSet_SGD_alpha_0.005_beta_5_L2_0_epoch_5.csv")

sgd_train1 <- sgd_train1[1:10000,"target"];sgd_train2 <- sgd_train2[1:10000,"target"]
sgd_test1  <- sgd_test1[1:10000,"target"]; sgd_test2  <- sgd_test2[1:10000,"target"]



##################################################################
#
# We can now train the 2nd stage model with the new meta features
#
#################################################################

### Combined the orginal features with the model predictions for training set
train_meta <- cbind(train,dataset_blend_train,sgd_train1,sgd_train2)

### Combined the orginal features with the model predictions for testing set
test_meta <- cbind(test,dataset_blend_test,sgd_test1,sgd_test2)

### Build the meta classifier (this needs to be tuned as well)
### Can also use several classifiers here and ensemble them together
xgtrain_meta = xgb.DMatrix(as.matrix(train_meta), label = y, missing = -9999)
gc()
watchlist <- list(train = xgtrain_meta)

param <- list(  objective = "binary:logistic", eta = 0.01, max_depth = 8,  
                subsample = 0.72, colsample_bytree = 0.74, eval_metric = "auc")

mod_meta <- xgb.train(params = param, data = xgtrain_meta, nrounds = 2, verbose = 1, watchlist = watchlist)

## Make final prediction
pred_meta <-predict(mod_meta, xgb.DMatrix(as.matrix(test_meta), missing = -9999), ntreelimit = mod_meta$bestInd)



