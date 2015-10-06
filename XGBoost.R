#--------- L I B R A R Y ------------------------------------------------

library(xgboost)
library(readr)
library(Ckmeans.1d.dp)
library(caret)

# -------- D A T A ---------------

cat("reading the train and test data\n")
path = "/Users/msegala/Documents/Personal/Kaggle/Springleaf/"

train <- read_csv(paste0(path, "train.csv", collapse = ""))
y <- train$target
ID <- train$ID
train <- train[,-c(1, 1934)]

test <- read_csv(paste0(path, "test.csv", collapse = ""))
test_ID <- test$ID
test <- test[,-1]

# Convert data columns into dobules
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



cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in 1:ncol(train)) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))    
  }
  #else{
  #  train[[f]] <- scale(train[[f]], center = TRUE, scale = TRUE)
  #  test[[f]]  <- scale(test[[f]],  center = TRUE, scale = TRUE)
  #}
}

cat("replacing missing values with -9999\n")
train[is.na(train)] <- -98989898
test[is.na(test)]   <- -98989898

## Look at near zero variance
#nearZeroVar(train, saveMetrics = TRUE)
nzv <- nearZeroVar(train)


#train_nzv <- train[, -nzv]
#test_nzv  <- test[, -nzv]
train <- train[, -nzv]
test  <- test[, -nzv]

#### SAVED AT THIS POINT........


### Find highly correlated features
#descrCor <- cor(train[1:100,1:100])
#descrCor_pear <-  cor(scale(train,center=TRUE,scale=TRUE), method="pearson")
#descrCor_spea <-  cor(scale(train,center=TRUE,scale=TRUE), method="spearman")
#corrplot(descrCor_pear, order = "hclust", mar=c(0,0,1,0), tl.pos="n", 
#         main="Pearson correlation of 100 sampled numerical features")
#corr_columns <- findCorrelation(descrCor_pear, cutoff = .99, verbose = F)
#ttt <- train[,-corr_columns]


### Write new data to file for other libraries
#train$target <- y
#train$ID <- ID
#train<-train[,c(ncol(train),1:(ncol(train)-1))]
#test$ID <- test_ID
#test<-test[,c(ncol(test),1:(ncol(test)-1))]
#write.csv(train,paste0(path, "train_new.csv", collapse = ""),row.names = FALSE)
#write.csv(test,paste0(path, "test_new.csv", collapse = ""),row.names = FALSE)


## Lets look at the columns with only one unique value.
col_ct = sapply(train, function(x) length(unique(x)))
cat("Constant feature count:", length(col_ct[col_ct==1]))


train.unique.count=lapply(train, function(x) length(unique(x)))
train.unique.count_1=unlist(train.unique.count[unlist(train.unique.count)==1])
train.unique.count_2=unlist(train.unique.count[unlist(train.unique.count)==2])
#train.unique.count_2=train.unique.count_2[-which(names(train.unique.count_2)=='target')]

delete_const=names(train.unique.count_1)
delete_NA56=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145175))
delete_NA89=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145142))
delete_NA918=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==144313))

print(length(c(delete_const,delete_NA56,delete_NA89,delete_NA918)))

#train=train[,!(names(train) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918))]
#test=test[,!(names(test) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918))]

train=train[,!(names(train) %in% c(delete_const))]
test=test[,!(names(test) %in% c(delete_const))]



####################################################
#
# Create interaction terms
#
####################################################

#interactions = c("0032","0437","0033","0906","0070","0003","0723","0591","0513","0216",
#                 "1330","0045","0452","0509","0491","0493","0044","0092","0352","0231","0081")


#"VAR_0003" "VAR_0794" "VAR_0070" "VAR_1328" "VAR_0136" "VAR_0003" "VAR_1126" "VAR_0969" "VAR_0885" "VAR_0504" "VAR_1790" "VAR_0087"
#"VAR_0809" "VAR_0880" "VAR_0852" "VAR_0854" "VAR_0086" "VAR_0238" "VAR_0704" "VAR_0539" "VAR_0201"

#interactions = c( 3,437,33,906,70,3,723,591,513,216,1330,45,452,509,491,493,44,92,352,231,81)
#interactions = colnames(train)[interactions]
interactions = c("VAR_0003", "VAR_0794", "VAR_0070", "VAR_1328", "VAR_0136", "VAR_0003", "VAR_1126", "VAR_0969", "VAR_0885", "VAR_0504",
                 "VAR_1790", "VAR_0087", "VAR_0809", "VAR_0880", "VAR_0852", "VAR_0854", "VAR_0086", "VAR_0238", "VAR_0704", "VAR_0539", "VAR_0201")

for (i in 1:length(interactions)){
  for (j in (i+1):length(interactions)){
  
    if (j <= length(interactions)){
      
      col1 = interactions[i]
      col2 = interactions[j]
      
      print(paste(col1,"_PLUS_",col2, sep=""))
      print(paste(col1,"_TIMES_",col2, sep="")
            )
      train[,paste(col1,"_PLUS_",col2, sep="")]  <- train[,col1] + train[,col2]
      train[,paste(col1,"_TIMES_",col2, sep="")] <- train[,col1] * train[,col2]    

      test[,paste(col1,"_PLUS_",col2, sep="")]  <- test[,col1] + test[,col2]
      test[,paste(col1,"_TIMES_",col2, sep="")] <- test[,col1] * test[,col2]    
      
    }    
  }
}

cat("replacing missing values with -9999\n")
train[is.na(train)] <- -98989898
test[is.na(test)]   <- -98989898



### Select Validation Set
set.seed(42)
val <- sample(1:nrow(train), 15000) #10% training data for validation

xgtrain = xgb.DMatrix(as.matrix(train[-val,]), label = y[-val], missing = -98989898)
xgval   = xgb.DMatrix(as.matrix(train[val,]), label = y[val], missing = -98989898)
#xgtrain = xgb.DMatrix(as.matrix(train), label = y, missing = -98989898)
#xgval   = xgb.DMatrix(as.matrix(train[val,]), label = y[val], missing = -98989898)
gc()

### reduce the memory footprint
#train=train[1:3,]
#gc()

watchlist <- list(eval = xgval, train = xgtrain)
#


#param <- list(  objective           = "binary:logistic", 
#                eta                 = 0.01,
#                max_depth           = 8,  # changed from default of 6
#                subsample           = 0.72,
#                colsample_bytree    = 0.74,
#                nthread             = 16,
#                eval_metric         = "auc"
#)

#clf <- xgb.train(   params              = param, 
#                    data                = xgtrain, 
#                    nrounds             = 1500, # changed from 500
#                    verbose             = 2, 
#                    early.stop.round    = 18,
#                    watchlist           = watchlist,
#                    maximize            = TRUE
#)

weight <- as.numeric(train[-val,2])*145232/145231
#sumwpos <- sum(weight * (y[-val]==1.0))
#sumwneg <- sum(weight * (y[-val]==0.0))
sumwpos <- sum((y==1.0))
sumwneg <- sum((y==0.0))
print(paste("weight statistics: wpos=", sumwpos, "wneg=", sumwneg, "ratio=", sumwneg / sumwpos))

#xgtrain = xgb.DMatrix(as.matrix(train[-val,]), label = y[-val], weight = weight, missing = NA)
#xgval   = xgb.DMatrix(as.matrix(train[val,]), label = y[val], weight = weight, missing = NA)
#gc()


#ETA - 0.3 nround = 60 to 190
#0.2 nround = 100 to 300
#0.1 nround = 250 to 800
#0.05 nround = 500 to 800
#0.01 nround = 1500 to 3000
#0.008 nround = 2500 to 3200
#0.0001 nround = 25000 to 35000

param <- list(  objective           = "binary:logistic", 
                eta                 = 0.01,
                max_depth           = 8,  # changed from default of 6
                min_child_weight    = 5,
                gamma               = 4,
                subsample           = 0.72,
                colsample_bytree    = 0.74,
                nthread             = 16,
                eval_metric         = "auc"
)


#param <- list(  objective           = "binary:logistic", 
#                booster             = 'gblinear',
#                eta                 = 0.02,
#                eval_metric         = "auc"
#)


#[1232]  eval-auc:0.797459	train-auc:0.941595
clf <- xgb.train(   params              = param, 
                    data                = xgtrain, 
                    #nrounds             = 1500, # changed from 500
                    nrounds             = 1500, # changed from 500
                    verbose             = 2, 
                    early.stop.round    = 18, #changed from 18
                    watchlist           = watchlist,
                    maximize            = TRUE
)

#xgb.save(clf, paste0(path, "xgboost.model", collapse = ""))

importance_matrix <- xgb.importance(model = clf)
print(importance_matrix[1:100])
top_features <- list(as.integer(importance_matrix[1:50,]$Feature))
xgb.plot.importance(importance_matrix[1:50,])
colnames(train)[as.integer(importance_matrix[1:50,]$Feature)]

### current best = eval-auc:0.788161  train-auc:0.998731
                    
xgtest <- xgb.DMatrix(as.matrix(test), missing = NA)

bst <- clf$bestInd
preds_out <- predict(clf, xgtest, ntreelimit = bst)

sub <- read_csv(paste0(path, "sample_submission.csv", collapse = ""))
sub$target <- preds_out
subversion <- 11
write_csv(sub, paste0(path, "test_submission_", subversion, ".csv", collapse = ""))

aaaa






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






##############
#
# Speed Test
#
###############

xgboost.time = list()
threads = c(1)
for (i in 1:length(threads)){
  thread = threads[i]
  xgboost.time[[i]] = system.time({
    
    weight <- as.numeric(train[1:10000,2])*145232/145231
    xgmat <- xgb.DMatrix(as.matrix(train[1:10000,]), label = y[1:10000], missing = NA,
                         weight = weight)

    param <- list(  objective           = "binary:logistic", 
                    eta                 = 0.03,
                    max_depth           = 6,  
                    eval_metric         = "auc"
                    scale_pos_weight    = 3.9,
                    eval_metric         = "ams@0.15"
                    #nthread             = thread
    )
        
    watchlist <- list("train" = xgmat)
    nround = 25
    print ("loading data end, start to boost trees")
    bst = xgb.train(param, xgmat, nround, watchlist );
    print ('finish training')
  })
}
xgboost.time








##########################################
#
# Parameter Tuning
#
##########################################

max_depth_        = c(10,20,50)
eta_              = c(0.001,0.01,0.1)
#nround_           = c(20,50,200)
nround_           = c(15)
gamma_            = c(0,10,100)
min_child_weight_ = c(0,10,1000)
subsample_        = c(0.7,0.9)
colsample_bytree_ = c(0.5,0.7)


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
                            silent = 1, nthread = 6, objective = 'binary:logistic', "eval_metric" = "auc")
              res <- xgb.cv(param, xgval, n, nfold=4, metrics={'auc'})
              
              best_train = res$train.auc.mean[n]
              best_test  = res$test.auc.mean[n]
              
              cat("iteration = ", i,": Max_Depth, Eta, NRound, Gamma, Min_Child_Weight, Subsample, ColSample = ",m,e,n,g,mi,s,c,"Best Train/Test = ",best_train,"/",best_test, "\n")
              
              df[i,] <- c(i,m,e,n,g,mi,s,c,best_train,best_test)
              i = i + 1              
              
            }
          }
        }
      }
    }
  }
}

df

### Best train
df[which.max( df[,9] ),]

### Best test
df[which.max( df[,10] ),]




nround <- 5
param <- list(max.depth=2,eta=1,silent=1,nthread = 2, objective='binary:logistic',"eval_metric" = "auc")
res <- xgb.cv(param, xgtrain, nround, nfold=5, metrics={'auc'})
res$test.auc.mean[nround]


