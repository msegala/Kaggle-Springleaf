## https://github.com/h2oai/h2o-2/tree/master/R/examples/Kaggle

#install.packages("h2o", repos=(c("http://s3.amazonaws.com/h2o-release/h2o/master/1555/R", getOption("repos"))))
#install.packages("h2o")

load("~/Documents/Personal/Kaggle/Springleaf/SaveMe.RData")

library(h2o)
library(stringr)
library(pROC)

## Connect to H2O server (On server(s), run 'java -Xmx8g -ea -jar h2o.jar -port 53322 -name TradeShift' first)
## Go to http://server:53322/ to check Jobs/Data/Models etc.
#h2oServer <- h2o.init(ip="server", port = 53322)
h2o.shutdown(h2oServer)
## Launch H2O directly on localhost, go to http://localhost:54321/ to check Jobs/Data/Models etc.!
#h2oServer <- h2o.init(nthreads = -1, max_mem_size = '8g')
h2oServer <- h2o.init(nthreads = -1)


##### MAKE IT SMALL FOR TESTING.....
#train = train[1:10000,]
#y = y[1:10000]
#test = test[1:10000,]

## Group variables
vars <- colnames(train)
#ID <- vars[1]
labels <- "target"
predictors <- vars
targets <- labels 

## Settings (at least one of the following two settings has to be TRUE)
validate = T #whether to compute CV error on train/validation split (or n-fold), potentially with grid search
submitwithfulldata = T #whether to use full training dataset for submission (if FALSE, then the validation model(s) will make test set predictions)

ensemble_size <- 1 # more -> lower variance
seed0 = 1337
reproducible_mode = T # Set to TRUE if you want reproducible results, e.g. for final Kaggle submission if you think you'll win :)  Note: will be slower for DL


## Attach the labels to the training data
trainWL <- cbind(train, y)
trainWL <- as.h2o(h2oServer, trainWL)
trainWL <- h2o.assign(trainWL, "trainWL")

testWL <- as.h2o(h2oServer, test)
testWL <- h2o.assign(testWL, "testWL")


# Split the training data into train/valid (90%/10%)
## Want to keep train large enough to make a good submission if submitwithfulldata = F
splits <- h2o.splitFrame(trainWL, ratios = 0.50)
train <- splits[[1]]
valid <- splits[[2]]

train <- h2o.assign(train, 'train')
valid <- h2o.assign(valid, 'valid')



cat("\nTraining H2O model on training/validation splits")
## Note: This could be grid search models, after which you would obtain the best model with model <- cvmodel@model[[1]]
cvmodel <- h2o.randomForest(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                            type="BigData", ntrees = 400, max_depth = 12, seed=seed0, min_rows = 22)


cvmodel <- h2o.gbm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                   distribution = "AUTO", type="BigData", ntrees = 100, max_depth = 3, learn_rate = 0.12, seed=seed0)




cvmodel <- h2o.gbm(training_frame=train, x=c(1:(ncol(train)-1)), y=ncol(train), nfolds = 3,
                   distribution = "AUTO", type="BigData", ntrees = 100, max_depth = 3, learn_rate = 0.12, seed=seed0)

cvmodel <- h2o.glm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                   family="gaussian", alpha = 0.43, lambda = 1e-02)


cvmodel <- h2o.deeplearning(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                            hidden=c(500,500), epochs=10, activation="RectifierWithDropout")

summary(cvmodel)


train_resp <- train[,ncol(train)] #actual label
train_preds <- h2o.predict(cvmodel, train)#[,3] #[,3] is probability for class 1
train_resp_df  <- as.data.frame(train_resp)
train_preds_df <- as.data.frame(train_preds)
cat("\nAUC on training data:", auc(as.numeric(train_resp_df$y),as.numeric(train_preds_df$predict)))

valid_resp <- valid[,ncol(train)]
valid_preds <- h2o.predict(cvmodel, valid)#[,3]
valid_resp_df  <- as.data.frame(valid_resp)
valid_preds_df <- as.data.frame(valid_preds)
cat("\nAUC on validation data:", auc(as.numeric(valid_resp_df$y),as.numeric(valid_preds_df$predict)))

######################################################
#
# Grid Search on CV set
#
######################################################

models <- c()
train_error <- c()
val_error <- c()

train_resp <- as.data.frame(train[,ncol(train)]) #actual label
valid_resp <- as.data.frame(valid[,ncol(train)]) #actual label

for (i in 1:8){

  print(paste("RUNNING:",i))
  
  if (i == 1){
    #AUC on training data: 0.7784455
    #AUC on validation data: 0.7447953
    dlmodel <- h2o.randomForest(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                                type="BigData", ntrees = 200, max_depth = 8, seed=seed0, balance_classes = F)  
  }
  else if (i == 2){
    #AUC on training data: 0.9726728
    #AUC on validation data: 0.758297
    dlmodel <- h2o.randomForest(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                                type="BigData", ntrees = 200, max_depth = 16, seed=seed0, balance_classes = F)  
  }
  else if (i == 3){
    #AUC on training data: 0.74786563
    #AUC on validation data: 0.7449874
    dlmodel <- h2o.randomForest(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                                type="BigData", ntrees = 700, max_depth = 8, seed=seed0, balance_classes = F)  
  }
  else if (i == 4){
    #AUC on training data: 0.9731295
    #AUC on validation data: 0.7604324
    dlmodel <- h2o.randomForest(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                                type="BigData", ntrees = 700, max_depth = 16, seed=seed0, balance_classes = F)  
  }
  else if (i == 5){
    #AUC on training data: 0.900409
    #AUC on validation data: 0.7593888
    dlmodel <- h2o.randomForest(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                                type="BigData", ntrees = 200, max_depth = 16, seed=seed0, min_rows = 24)  
  }
  else if (i == 6){
    #AUC on training data: 0.8404489
    #AUC on validation data: 0.7559922
    dlmodel <- h2o.randomForest(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                                type="BigData", ntrees = 1000, max_depth = 12, seed=seed0, min_rows = 34)  
  }
  else if (i == 7){
    #AUC on training data: 0.7602488
    #AUC on validation data: 0.7404266
    dlmodel <- h2o.randomForest(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                                type="BigData", ntrees = 1000, max_depth = 7, seed=seed0, min_rows = 22)  
  }
  else if (i == 8){
    #AUC on training data: 0.8551222
    #AUC on validation data: 0.7565042
    dlmodel <- h2o.randomForest(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                                type="BigData", ntrees = 400, max_depth = 12, seed=seed0, min_rows = 22)  
  }
  
  
  else if (i == 9){
    #AUC on training data: 0.7323531
    #AUC on validation data: 0.7270222
    dlmodel <- h2o.glm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                       family="gaussian", alpha = 0.43, lambda = 1e-02)  
  }
  else if (i == 10){
    #AUC on training data: 0.7604456
    #AUC on validation data:  0.745433
    dlmodel <- h2o.glm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                       family="gaussian", alpha = 0.43, lambda = 1e-05)  
  }
  else if (i == 11){
    #AUC on training data: AUC on training data: 0.7628347
    #AUC on validation data:AUC on validation data: 0.7466417
    dlmodel <- h2o.glm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                       family="gaussian", alpha = 0.43, lambda = 1e-07)  
  }
  else if (i == 12){
    #AUC on training data: AUC on training data: 0.7626303
    #AUC on validation data: AUC on validation data: 0.7464264  
    dlmodel <- h2o.glm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                       family="gaussian", alpha = 0.73, lambda = 1e-07)  
  }
  else if (i == 13){
    #AUC on training data: AUC on training data: 0.7638
    #AUC on validation data: AUC on validation data: 0.74768
    dlmodel <- h2o.glm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                       family="gaussian", alpha = 0, lambda = 1e-07)  
  }
  else if (i == 14){
    #AUC on training data: AUC on training data: 0.7624796
    #AUC on validation data: AUC on validation data: 0.7462628
    dlmodel <- h2o.glm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                       family="gaussian", alpha = 1, lambda = 1e-07)  
  }
  

  
  else if (i == 15){
    #AUC on training data: AUC on training data: 0.7785576
    #AUC on validation data: AUC on validation data: 0.7609099
    dlmodel <- h2o.gbm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                       distribution = "AUTO", type="BigData", ntrees = 100, max_depth = 3, learn_rate = 0.12)
  }
  else if (i == 16){
    #AUC on training data: AUC on training data: 0.9891246
    #AUC on validation data: AUC on validation data: 0.7574609
    dlmodel <- h2o.gbm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                       distribution = "AUTO", type="BigData", ntrees = 100, max_depth = 10, learn_rate = 0.12)
  }
  else if (i == 17){
    #AUC on training data: AUC on training data: 0.9017845
    #AUC on validation data: AUC on validation data: 0.7541702
    dlmodel <- h2o.gbm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                       distribution = "AUTO", type="BigData", ntrees = 100, max_depth = 10, learn_rate = 0.012)
  }
  else if (i == 18){
    #AUC on training data: AUC on training data:
    #AUC on validation data: AUC on validation data: 
    dlmodel <- h2o.gbm(training_frame=train, validation_frame=valid, x=c(1:(ncol(train)-1)), y=ncol(train),
                       distribution = "AUTO", type="BigData", ntrees = 1000, max_depth = 10, learn_rate = 0.012)
  }
  
  
  
  
  #models <- c(models, dlmodel)
    
  train_preds <- as.data.frame(h2o.predict(dlmodel, train))
  auc_train <- auc(as.numeric(train_resp$y),as.numeric(train_preds$predict))
  train_error <- c(train_error,auc_train)
  cat("\nAUC on training data:", auc_train)
  
  valid_preds <- as.data.frame(h2o.predict(dlmodel, valid))
  auc_val <- auc(as.numeric(valid_resp$y),as.numeric(valid_preds$predict))
  val_error <- c(val_error, auc_val)
  cat("\nAUC on validation data:", auc_val)
  cat("\n")
}




######################################################
#
# Now that CV is set, train on whole dataset
#
######################################################

p <- cvmodel@allparameters

## Build an ensemble model on full training data - should perform better than the CV model above
for (n in 1:ensemble_size) {
  cat("\n\nBuilding ensemble model", n, "of", ensemble_size, "...\n")
  model <-  h2o.randomForest(training_frame=trainWL,  
                             key = paste0("cv_ensemble_", n, "_of_", ensemble_size),
                             x=c(1:(ncol(trainWL)-1)), y=ncol(trainWL),
                             type="BigData", 
                             ntree = p$ntrees, depth = p$max_depth, seed=p$seed + n, verbose = T)
            
  ## Aggregate ensemble model predictions
  test_preds <- h2o.predict(model, testWL)
  if (n == 1) {
    test_preds_blend <- test_preds
  } else {
    test_preds_blend <- cbind(test_preds_blend, test_preds[,1])
  }
}


## Now create submission
cat (paste0("\n Number of ensemble models: ", ncol(test_preds_blend)))
ensemble_average <- matrix("ensemble_average", nrow = nrow(test_preds_blend), ncol = 1)
ensemble_average <- rowMeans(as.data.frame(test_preds_blend)) # Simple ensemble average, consider blending/stacking
ensemble_average <- as.data.frame(ensemble_average)
print(head(ensemble_average))

sub <- read_csv(paste0(path, "sample_submission.csv", collapse = ""))
sub$target <- ensemble_average$ensemble_average









