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

load("~/Documents/Personal/Kaggle/Springleaf/SaveMe.RData")


##### MAKE IT SMALL FOR TESTING.....
train = train[1:10000,]
y = y[1:10000]
test = test[1:10000,]

set.seed(42)
val <- sample(1:nrow(train), 1000) #10% training data for validation


# -------- h2o -------------------

h2o.shutdown(h2oServer)
## Launch H2O directly on localhost, go to http://localhost:54321/ to check Jobs/Data/Models etc.!
h2oServer <- h2o.init(nthreads = -1, max_mem_size = '8g')

## Attach the labels to the training data
#trainWL <- cbind(train, y)
#trainWL <- as.h2o(h2oServer, trainWL)
#trainWL <- h2o.assign(trainWL, "trainWL")
testWL <- as.h2o(h2oServer, test)
testWL <- h2o.assign(testWL, "testWL")


# -------- XGBoost -------------------

#xgtrain = xgb.DMatrix(as.matrix(train), label = y, missing = -98989898)
#gc()


# -------- Parameters ----------------
n_folds = 5
verbose = TRUE
shuffle = FALSE


# -------- Setup Stacking ------------

### Returns train inidices for n_folds using StratifiedKFold
skf = createFolds(y, k = n_folds , list = TRUE, returnTrain = TRUE)

### Create a list of models to run
clfs  <- c("xgboost1","xgboost2","rf1")
types <- c("xgboost","xgboost","h2o")

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
      xgtrain_tmp = xgb.DMatrix(as.matrix(X_train), label = y_train, missing = -98989898)
      xgval_tmp   = xgb.DMatrix(as.matrix(X_test),  label = y_test,  missing = -98989898)
      gc()
      watchlist <- list(eval = xgval_tmp, train = xgtrain_tmp)
      
      if (clf == "xgboost1"){
        if(i==1){cat("Running xgboost1....\n")}
        param <- list(  objective = "binary:logistic", eta = 0.01, max_depth = 8,  
                        subsample = 0.72, colsample_bytree = 0.74, nthread = 16, eval_metric = "auc")
        
        mod <- xgb.train(   params = param, data = xgtrain_tmp, nrounds = 2, verbose = 1, 
                            early.stop.round = 18, watchlist = watchlist, maximize = TRUE)
      }
      else if (clf == "xgboost2"){
        if(i==1){cat("Running xgboost2....\n")}
        param <- list(  objective = "binary:logistic", eta = 0.1, max_depth = 18,  
                        subsample = 0.72, colsample_bytree = 0.74, nthread = 16, eval_metric = "auc")
        
        mod <- xgb.train(   params = param, data = xgtrain_tmp, nrounds = 2, verbose = 2, 
                            early.stop.round = 18, watchlist = watchlist, maximize = TRUE)
      }
      
      rm(xgtrain_tmp)
      rm(xgval_tmp)
                  
      ### Predict the probability of current folds test set and store results.
      dataset_blend_train[-tmp_train, j] <- predict(mod, xgb.DMatrix(as.matrix(X_test), missing = -98989898), 
                                                    ntreelimit = mod$bestInd)

      ### Predict the probabilty for the true teest set and store results
      dataset_blend_test_j[, i] <- predict(mod, xgb.DMatrix(as.matrix(test), missing = -98989898), 
                                           ntreelimit = mod$bestInd)
    }
    else if (tmp_type == "h2o"){
    
      if(i==1){cat("Running an H2o RF model....\n")}
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
      
      ### Predict the probability of current folds test set and store results.
      dataset_blend_train[-tmp_train, j] <- as.data.frame(h2o.predict(mod, X_testWL))[,1]
      
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




##################################################################
#
# We can now train the 2nd stage model with the new meta features
#
#################################################################

### Combined the orginal features with the model predictions for training set
train_meta <- cbind(train,dataset_blend_train)

### Combined the orginal features with the model predictions for testing set
test_meta <- cbind(test,dataset_blend_test)

### Build the meta classifier (this needs to be tuned as well)
### Can also use several classifiers here and ensemble them together
xgtrain_meta = xgb.DMatrix(as.matrix(train_meta), label = y, missing = -98989898)
gc()
watchlist <- list(train = xgtrain_meta)

param <- list(  objective = "binary:logistic", eta = 0.01, max_depth = 8,  
                subsample = 0.72, colsample_bytree = 0.74, nthread = 16, eval_metric = "auc")

mod_meta <- xgb.train(params = param, data = xgtrain_meta, nrounds = 2, verbose = 1, watchlist = watchlist)

## Make final prediction
pred_meta <-predict(mod_meta, xgb.DMatrix(as.matrix(test_meta), missing = -98989898), ntreelimit = mod_meta$bestInd)


