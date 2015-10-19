

##############
#
# Bagging Classifier
# run several times the same algorithm, 
# with random selection of observations and features, and take the average of the output.
#
##############

bagged_prediction <- data.frame(matrix(NA, nrow = nrow(test), ncol = 5))

i <- 1
for (i in 1:5){
  print(i)
  #set.seed(42*i + i*13)
  
  train_bag <- sample(1:nrow(train), 116185) #80% training data 
  columns_bag <- sample(1:ncol(train), 1197) #80% columns
  
  ### Create tmp training and y by selecting only the bagged training rows and columns
  tmp    <- train[train_bag,columns_bag]
  y_tmp  <- y[train_bag]
  
  val <- sample(1:nrow(tmp), 11618) #10% training data for validation
  
  xgtrain = xgb.DMatrix(as.matrix(tmp[-val,]), label = y_tmp[-val], missing = -98989898)
  xgval   = xgb.DMatrix(as.matrix(tmp[val,]),  label = y_tmp[val],  missing = -98989898)
  gc()
  
  rm(tmp)
  rm(y_tmp)
  
  watchlist <- list(eval = xgval, train = xgtrain)
  
  param <- list(  objective           = "binary:logistic", 
                  eta                 = 0.01,
                  max_depth           = 8,  # changed from default of 6
                  subsample           = 0.92,
                  colsample_bytree    = 0.94,
                  nthread             = 16,
                  eval_metric         = "auc")
  
  clf <- xgb.train(   params              = param, 
                      data                = xgtrain, 
                      nrounds             = 1000, 
                      verbose             = 2, 
                      early.stop.round    = 8,
                      watchlist           = watchlist,
                      maximize            = TRUE)
  
  xgtest <- xgb.DMatrix(as.matrix(test), missing = NA)
  bst <- clf$bestInd
  preds_out <- predict(clf, xgtest, ntreelimit = bst)
  
  bagged_prediction[,i] <- preds_out  
  
}

bagged_prediction$avg <- rowMeans(bagged_prediction) 
bagged_prediction[1:5,]

sub <- read_csv(paste0(path, "sample_submission.csv", collapse = ""))
sub$target <- bagged_prediction$avg
write_csv(sub, paste0(path, "test_submission_bagged_ensemble_train_80_cols_80.csv", collapse = ""))
