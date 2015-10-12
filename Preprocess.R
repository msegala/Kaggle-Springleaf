#--------- L I B R A R Y ------------------------------------------------

library(xgboost)
library(readr)
library(Ckmeans.1d.dp)
library(caret)

# -------- D A T A ---------------

cat("reading the train and test data\n")
path = "/Users/msegala/Documents/Personal/Kaggle/Kaggle-Springleaf/"

train <- read_csv(paste0(path, "train.csv", collapse = ""))
y <- train$target
ID <- train$ID
train <- train[,-c(1, 1934)]

test <- read_csv(paste0(path, "test.csv", collapse = ""))
test_ID <- test$ID
test <- test[,-1]

cat("replacing missing values with -98989898\n")
train[is.na(train)] <- -98989898
test[is.na(test)]   <- -98989898


# ------ Cleaning --------------

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
#### Check for constant columns and remove them
#
col_ct = sapply(train, function(x) length(unique(x)))
cat("Constant feature count:", length(col_ct[col_ct==1]))
train = train[, !names(train) %in% names(col_ct[col_ct==1])]
test  = test[,  !names(test)  %in% names(col_ct[col_ct==1])]


#
#### Look at near zero variance
#
nzv <- nearZeroVar(train)
nzv_cols <- colnames(train)[nzv]


#
#### Split data into numerical and categorical data.frames
#
train_numr = train[, sapply(train, is.numeric)]
train_char = train[, sapply(train, is.character)]
cat("Numerical column count : ", dim(train_numr)[2], "; Character column count : ", dim(train_char)[2])


#
#### Find highly correlated features in numerical data.frame
#
descrCor_pear <-  cor(scale(train_numr,center=TRUE,scale=TRUE), method="pearson")
descrCor_spea <-  cor(scale(train_numr,center=TRUE,scale=TRUE), method="spearman")
#corrplot(descrCor_pear, order = "hclust", mar=c(0,0,1,0), tl.pos="n", main="Pearson correlation of 100 sampled numerical features")
corr_columns_pear <- findCorrelation(descrCor_pear, cutoff = .95, verbose = F)
corr_columns_spea <- findCorrelation(descrCor_spea, cutoff = .95, verbose = F)
corr_in_both <- intersect(corr_columns_pear,corr_columns_spea)
corr_in_both_cols <- colnames(train_numr)[corr_in_both]


#
#### Remove highly correlated features in numeric data.frame and perfrom feature importance in XGBoost
#
train_numr_reduced <- train_numr[,-corr_in_both]

set.seed(42)
val <- sample(1:nrow(train_numr_reduced), 15000) #10% training data for validation
xgtrain_numr = xgb.DMatrix(as.matrix(train_numr_reduced[-val,]), label = y[-val], missing = -98989898)
xgval_numr   = xgb.DMatrix(as.matrix(train_numr_reduced[val,]),  label = y[val],  missing = -98989898)
gc()
watchlist <- list(eval = xgval_numr, train = xgtrain_numr)

param <- list(  objective           = "binary:logistic", 
                eta                 = 0.01, max_depth           = 8,  subsample           = 0.72,
                colsample_bytree    = 0.74, nthread             = 16, eval_metric         = "auc")

clf_numr <- xgb.train(   params              = param,     data                = xgtrain_numr, 
                         nrounds             = 500,       verbose             = 2, early.stop.round    = 18,
                         watchlist           = watchlist, maximize            = TRUE)


importance_matrix <- xgb.importance(model = clf_numr)
print(importance_matrix[1:100])
top_features <- list(as.integer(importance_matrix[1:50,]$Feature))
xgb.plot.importance(importance_matrix[1:50,])
colnames(train_numr_reduced)[as.integer(importance_matrix[1:50,]$Feature)]

## Top 50 variables
#"VAR_0068" "VAR_0794" "VAR_0072" "VAR_0129" "VAR_1327" "VAR_0070" "VAR_0074" "VAR_1126" "VAR_0003"
#"VAR_0969" "VAR_0504" "VAR_1746" "VAR_0885" "VAR_1790" "VAR_0880" "VAR_0807" "VAR_0539" "VAR_0707"
#"VAR_0852" "VAR_0234" "VAR_0706" "VAR_0854" "VAR_1397" "VAR_0211" "VAR_1742" "VAR_1410" "VAR_0002"
#"VAR_1135" "VAR_0966" "VAR_1377" "VAR_0612" "VAR_0884" "VAR_0711" "VAR_0806" "VAR_0225" "VAR_0067"
#"VAR_0805" "VAR_1127" "VAR_0271" "VAR_0870" "VAR_0577" "VAR_0761" "VAR_0542" "VAR_0879" "VAR_0502"
#"VAR_0709" "VAR_0278" "VAR_1113" "VAR_0883"


#
#### Generate new training/testing sets
#

##--- Remove NZV
train_nzv <- train[ , !names(train) %in% nzv_cols]
test_nzv  <- test[  , !names(test)  %in% nzv_cols]

##--- Remove Highly Correlated
train_corr <- train[ , !names(train) %in% corr_in_both_cols]
test_corr  <- test[  , !names(test)  %in% corr_in_both_cols]

##--- Remove NZV AND Highly Correlated
train_nzv_corr <- train[ , !names(train) %in% nzv_cols]
test_nzv_corr  <- test[  , !names(test)  %in% nzv_cols]
train_nzv_corr <- train_nzv_corr[ , !names(train_nzv_corr) %in% corr_in_both_cols]
test_nzv_corr  <- test_nzv_corr[  , !names(test_nzv_corr)  %in% corr_in_both_cols]




#
#### Make interactions of top 20
#
interactions = c("VAR_0068", "VAR_0794", "VAR_0072", "VAR_0129", "VAR_1327", "VAR_0070", "VAR_0074", "VAR_1126", "VAR_0003", "VAR_0969",
                 "VAR_0504", "VAR_1746", "VAR_0885", "VAR_1790", "VAR_0880", "VAR_0807", "VAR_0539", "VAR_0707", "VAR_0852", "VAR_0234")

train_interaction_tmp <- train[ , names(train) %in% interactions]
test_interaction_tmp  <- test[ ,  names(test) %in% interactions]

### reaplce negatives with Nan and then do it. This will prevent -98989898 values
### multpying real values
train_interaction_tmp[train_interaction_tmp <0 ] <- NaN
test_interaction_tmp[test_interaction_tmp < 0]   <- NaN

train_interactions <- data.frame(matrix(NA, nrow = nrow(train), ncol = 0))
test_interactions  <- data.frame(matrix(NA, nrow = nrow(test), ncol = 0))

for (i in 1:length(interactions)){
  for (j in (i+1):length(interactions)){
    
    if (j <= length(interactions)){
      
      col1 = interactions[i]
      col2 = interactions[j]
      
      print(paste(col1,"_PLUS_",col2, sep=""))
      print(paste(col1,"_TIMES_",col2, sep=""))
      
      train_interactions[,paste(col1,"_PLUS_",col2, sep="")]  <- train_interaction_tmp[,col1] + train_interaction_tmp[,col2]
      train_interactions[,paste(col1,"_TIMES_",col2, sep="")] <- train_interaction_tmp[,col1] * train_interaction_tmp[,col2]    
      
      test_interactions[,paste(col1,"_PLUS_",col2, sep="")]  <- test_interaction_tmp[,col1] + test_interaction_tmp[,col2]
      test_interactions[,paste(col1,"_TIMES_",col2, sep="")] <- test_interaction_tmp[,col1] * test_interaction_tmp[,col2]    
    }    
  }
}
train_interactions[is.na(train_interactions)] <- -98989898
test_interactions[is.na(test_interactions)]   <- -98989898

rm(train_interaction_tmp);rm(test_interaction_tmp)


#
#### Convert categorical columns into factors
#

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in 1:ncol(train_nzv)) {
  if (class(train_nzv[[f]])=="character") {
    levels <- unique(c(train_nzv[[f]], test_nzv[[f]]))
    train_nzv[[f]] <- as.integer(factor(train_nzv[[f]], levels=levels))
    test_nzv[[f]]  <- as.integer(factor(test_nzv[[f]],  levels=levels))    
  }
}

for (f in 1:ncol(train_corr)) {
  if (class(train_corr[[f]])=="character") {
    levels <- unique(c(train_corr[[f]], test_corr[[f]]))
    train_corr[[f]] <- as.integer(factor(train_corr[[f]], levels=levels))
    test_corr[[f]]  <- as.integer(factor(test_corr[[f]],  levels=levels))    
  }
}

for (f in 1:ncol(train_nzv_corr)) {
  if (class(train_nzv_corr[[f]])=="character") {
    levels <- unique(c(train_nzv_corr[[f]], test_nzv_corr[[f]]))
    train_nzv_corr[[f]] <- as.integer(factor(train_nzv_corr[[f]], levels=levels))
    test_nzv_corr[[f]]  <- as.integer(factor(test_nzv_corr[[f]],  levels=levels))    
  }
}


#
#### Save all the final dataframes and write to file
#

save.image("~/Documents/Personal/Kaggle/Kaggle-Springleaf/DataProccessing.RData")

# Save NZV
train_nzv$target <- y
train_nzv$ID <- ID
train_nzv<-train_nzv[,c(ncol(train_nzv),1:(ncol(train_nzv)-1))]
test_nzv$ID <- test_ID
test_nzv<-test_nzv[,c(ncol(test_nzv),1:(ncol(test_nzv)-1))]
write.csv(train_nzv,paste0(path, "train_nzv.csv", collapse = ""),row.names = FALSE)
write.csv(test_nzv,paste0(path, "test_nzv.csv", collapse = ""),row.names = FALSE)

# Save Corr
train_corr$target <- y
train_corr$ID <- ID
train_corr<-train_corr[,c(ncol(train_corr),1:(ncol(train_corr)-1))]
test_corr$ID <- test_ID
test_corr<-test_corr[,c(ncol(test_corr),1:(ncol(test_corr)-1))]
write.csv(train_corr,paste0(path, "train_corr.csv", collapse = ""),row.names = FALSE)
write.csv(test_corr,paste0(path, "test_corr.csv", collapse = ""),row.names = FALSE)

# Save NZV + Corr
train_nzv_corr$target <- y
train_nzv_corr$ID <- ID
train_nzv_corr<-train_nzv_corr[,c(ncol(train_nzv_corr),1:(ncol(train_nzv_corr)-1))]
test_nzv_corr$ID <- test_ID
test_nzv_corr<-test_nzv_corr[,c(ncol(test_nzv_corr),1:(ncol(test_nzv_corr)-1))]
write.csv(train_nzv_corr,paste0(path, "train_nzv_corr.csv", collapse = ""),row.names = FALSE)
write.csv(test_nzv_corr,paste0(path, "test_nzv_corr.csv", collapse = ""),row.names = FALSE)

# Save Interactions, can add these later to the dataframes
write.csv(train_interactions,paste0(path, "train_interactions.csv", collapse = ""),row.names = FALSE)
write.csv(test_interactions,paste0(path, "test_interactions.csv", collapse = ""),row.names = FALSE)

