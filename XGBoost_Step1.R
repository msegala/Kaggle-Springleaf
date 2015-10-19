#--------- L I B R A R Y ------------------------------------------------

library(xgboost)
library(readr)
library(Ckmeans.1d.dp)
library(caret)
library(doParallel)

# -------- D A T A ---------------

cat("reading the train and test data\n")
path = "/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/"

#train <- read_csv(paste0(path, "train_nzv_corr.csv", collapse = ""))
#train <- read_csv(paste0(path, "train_nzv.csv", collapse = ""))
train <- read_csv(paste0(path, "train.csv", collapse = ""))
y <- train$target
ID <- train$ID
train <- train[,-c(1, length(train))]

#test <- read_csv(paste0(path, "test_nzv_corr.csv", collapse = ""))
#test <- read_csv(paste0(path, "test_nzv.csv", collapse = ""))
test <- read_csv(paste0(path, "test.csv", collapse = ""))
test_ID <- test$ID
test <- test[,-1]


##### If using base train/test fix it up a bit
useFull = TRUE
if(useFull){
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
}


#
#### Add interaction terms
#
addInteractions = TRUE
if(addInteractions){
  train_interactions <- read_csv(paste0(path, "train_interactions.csv", collapse = ""))
  test_interactions  <- read_csv(paste0(path, "test_interactions.csv", collapse = ""))
  
  train <- cbind(train, train_interactions)
  test  <- cbind(test,  test_interactions)
}

cat("replacing missing values with -9999\n")
train[is.na(train)] <- -9999
test[is.na(test)]   <- -9999

train[train<0] <- -9999
test[test<0]   <- -9999


set.seed(222)
val <- sample(1:nrow(train), round(0.3*nrow(train))) #30% training data for validation
xgtrain = xgb.DMatrix(as.matrix(train[-val,]), label = y[-val], missing = -9999)
xgval   = xgb.DMatrix(as.matrix(train[val,]),  label = y[val],  missing = -9999)
gc()
watchlist <- list(eval = xgval, train = xgtrain)

#param <- list(  objective           = "binary:logistic", 
#                eta                 = 0.005, max_depth           = 10,  subsample           = 1.0,
#                colsample_bytree    = 0.352, eval_metric         = "auc")

#param <- list(  objective           = "binary:logistic", 
#                eta                 = 0.005, max_depth           = 12,  subsample           = 1.0,
#                colsample_bytree    = 0.3, eval_metric         = "auc")

#param <- list(  objective           = "binary:logistic", 
#                eta                 = 0.005, max_depth           = 14,  subsample           = 1.0,
#                colsample_bytree    = 0.3, eval_metric         = "auc",
#                min_child_weight    = 6, gamma               = 6)

param <- list(  objective           = "binary:logistic", 
                eta                 = 0.01, max_depth           = 9,  subsample   = 0.7,
                colsample_bytree    = 0.5, eval_metric         = "auc",
                min_child_weight    = 6, alpha               = 4, gamma   = 2,
                nthread=16)


start <- proc.time()
clf <- xgb.train(   params    = param,     data     = xgtrain, early.stop.round    = 100,
                    nrounds   = 5000,      verbose  = 1,  print.every.n = 10, 
                    watchlist = watchlist, maximize = TRUE)
tot <- proc.time() - start
tot

bst <- clf$bestInd

#### For NZV with eta=0.005, depth=10,cols=0.352, na = -9999
#[2941]  eval-auc:0.791248  train-auc:0.998462
#Stopping. Best iteration: 2920

#### For Base Data with eta=0.005, depth=10,cols=0.352, na = -9999
#Stopping. Best iteration: 2545

#### For Base Data with eta=0.005, depth=12,cols=0.3, na = -9999
#Stopping. Best iteration: 1979

#### For Base Data with eta=0.005, depth=12,cols=0.3, min_child_weight= 5, gamma= 4, na = -9999
#Stopping. Best iteration: 2379

#### For Base Data + Interactiosn with eta=0.005, depth=12,cols=0.3, min_child_weight= 5, gamma= 4, na = -9999
#Stopping. Best iteration: 2519

#### For Base Data with eta=0.005, depth=14,cols=0.3, min_child_weight= 6, gamma= 6, na = -9999
#Stopping. Best iteration: 2589

#### For Base Data with eta=0.01, depth=9,cols=0.5, min_child_weight=6, subsample=0.7, alpha=4, na = -9999
#Stopping. Best iteration: 2884

#### For Base Data with eta=0.01, depth=9,cols=0.5, min_child_weight=6, subsample=0.7, alpha=5, na = -9999
#Stopping. Best iteration: 2866

#### For Base Data with eta=0.01, depth=9,cols=0.5, min_child_weight=6, subsample=0.7, alpha=3, na = -9999
#Stopping. Best iteration: 2790

#### For Base Data with eta=0.01, depth=9,cols=0.5, min_child_weight=5, subsample=0.7, alpha=3, na = -9999
#Stopping. Best iteration: 2336

#### For Base Data with eta=0.01, depth=10,cols=0.5, min_child_weight=7, subsample=0.7, alpha=4, na = -9999
#Stopping. Best iteration: 2403

#### For Base Data with eta=0.01, depth=10,cols=0.5, min_child_weight=6, subsample=0.7, alpha=4, na = -9999
#Stopping. Best iteration: 2003

#### For Base Data with eta=0.01, depth=9,cols=0.5, min_child_weight=6, subsample=0.7, alpha=4, gamma=2, na = -9999
#Stopping. Best iteration: 2484

### Train on full dataset
xgtrain = xgb.DMatrix(as.matrix(train), label = y, missing = -9999)
watchlist <- list(train = xgtrain)

start <- proc.time()
clf <- xgb.train(   params    = param,     data     = xgtrain,
                    nrounds   = bst,       verbose  = 1, print.every.n = 10,    
                    watchlist = watchlist)
tot <- proc.time() - start
tot


xgtest <- xgb.DMatrix(as.matrix(test), missing = -9999)
preds_out <- predict(clf, xgtest)

sub <- read_csv(paste0(path, "sample_submission.csv", collapse = "")) 
sub$target <- preds_out
subversion <- 1
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_NZV_FullTrain_Eta_0.005_Depth_10_Cols_0.352_Version_", subversion, ".csv", collapse = ""))
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_WithInteractions_FullTrain_Eta_0.005_Depth_10_Cols_0.352_Version_", subversion, ".csv", collapse = ""))
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.005_Depth_10_Cols_0.352_Version_", subversion, ".csv", collapse = ""))
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.0025_Depth_10_Cols_0.3_Version_", subversion, ".csv", collapse = ""))

#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.005_Depth_12_Cols_0.3_Version_", subversion, ".csv", collapse = ""))
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.005_Depth_12_Cols_0.3_MinChildWeight_5_Gamma_4_Version_", subversion, ".csv", collapse = ""))

#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_WithInteractions_FullTrain_Eta_0.005_Depth_12_Cols_0.3_MinChildWeight_5_Gamma_4_Version_", subversion, ".csv", collapse = ""))
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.005_Depth_14_Cols_0.3_MinChildWeight_6_Gamma_6_Version_", subversion, ".csv", collapse = ""))



#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.01_Depth_9_Cols_0.5_SubSample_0.7_MinChildWeight_6_Alpha_4_Version_", subversion, ".csv", collapse = ""))
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.01_Depth_9_Cols_0.5_SubSample_0.7_MinChildWeight_6_Alpha_5_Version_", subversion, ".csv", collapse = ""))
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.01_Depth_9_Cols_0.5_SubSample_0.7_MinChildWeight_6_Alpha_3_Version_", subversion, ".csv", collapse = ""))
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.01_Depth_9_Cols_0.5_SubSample_0.7_MinChildWeight_5_Alpha_3_Version_", subversion, ".csv", collapse = ""))
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.01_Depth_10_Cols_0.5_SubSample_0.7_MinChildWeight_7_Alpha_4_Version_", subversion, ".csv", collapse = ""))

# still need to submit on own
#write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.01_Depth_10_Cols_0.5_SubSample_0.7_MinChildWeight_6_Alpha_4_Version_", subversion, ".csv", collapse = ""))
write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_FullTrain_Eta_0.01_Depth_9_Cols_0.5_SubSample_0.7_MinChildWeight_6_Alpha_4_Gamma_2_Version_", subversion, ".csv", collapse = ""))






#
#### Boost from initial prediction (THIS DIDNT WORK.....)
#
### Started at 0.994093
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=TRUE, will always give you margin values before logistic transformation
ptrain <- predict(clf, xgtrain, outputmargin=TRUE)
# set the base_margin property of dtrain base margin is the base prediction we will boost from
setinfo(xgtrain, "base_margin", ptrain)
clf_boosted <- xgb.train( params    = param,     data     = xgtrain,
                          nrounds   = 500,       verbose  = 1,   
                          watchlist = watchlist )

bst <- clf_boosted$bestInd
preds_out <- predict(clf_boosted, xgtest, ntreelimit = bst)

sub <- read_csv(paste0(path, "sample_submission.csv", collapse = "")) 
sub$target <- preds_out
subversion <- 1
write_csv(sub, paste0(path, "/output/XGBoost/submission_XGBoost_FullData_WithInteractions_FullTrain_Eta_0.005_Depth_10_Cols_0.352_BOOSTED_500_Version_", subversion, ".csv", collapse = ""))





















##########################################
#
# Parameter Tuning
#
##########################################

max_depth_        = c(6,8,10)
eta_              = c(0.1,0.05)
nround_           = c(1000)
gamma_            = c(0)
min_child_weight_ = c(1)
subsample_        = c(0.72)
colsample_bytree_ = c(0.94,0.74)

df = data.frame(i                = numeric(),
                max_depth        = numeric(),
                eta              = numeric(),
                nround           = numeric(),
                gamma            = numeric(),
                min_child_weight = numeric(),
                subsample        = numeric(),
                colsample        = numeric(),
                best_train       = numeric(),
                best_test        = numeric())

set.seed(42)
#val <- sample(1:nrow(train), 15000) #10% training data for validation
val <- sample(1:nrow(train), 70000) #50% training data for validation
xgtrain = xgb.DMatrix(as.matrix(train[-val,]), label = y[-val], missing = -98989898)
xgval   = xgb.DMatrix(as.matrix(train[val,]),  label = y[val],  missing = -98989898)
gc()
watchlist <- list(eval = xgval, train = xgtrain)

i = 1
for (m in max_depth_){
  for (e in eta_){
    for (n in nround_){
      for (g in gamma_){
        for (mi in min_child_weight_){
          for (s in subsample_){
            for (c in colsample_bytree_){
              
              param <- list(max.depth        = m,
                            eta              = e,
                            gamma            = g,
                            min_child_weight = mi,
                            subsample        = s,
                            colsample_bytree = c,                            
                            silent = 1, objective = 'binary:logistic', "eval_metric" = "auc")
              
              clf <- xgb.train(   params              = param,  data               = xgtrain, 
                                  nrounds             = n,      verbose            = 1,   early.stop.round    = 18,
                                  watchlist           = watchlist, maximize           = TRUE)
              
              best_train = clf$bestScore
              best_test  = clf$bestScore
              
              cat("iteration = ", i,": Max_Depth, Eta, NRound, Gamma, Min_Child_Weight, Subsample, ColSample = ",m,e,n,g,mi,s,c,"Best Train/Test = ",best_train,"/",best_test, "\n")
              
              df[i,] <- c(i,m,e,n,g,mi,s,c,best_train,best_test)
              i = i + 1              
              
              print(df)
              
            }
          }
        }
      }
    }
  }
}

df


### Best test
df[which.max( df[,10] ),]


