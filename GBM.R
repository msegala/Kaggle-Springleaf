
require(gbm)
require(glmnet)
require(Metrics)

x_train = train[-val,]
x_val = train[val,]

y_train = y[-val]
y_val = y[val]

x_train$y <- y_train
x_val$y <- y_val

### Just put y in first column
col_idx <- grep("y", names(x_train))
x_train <- x_train[, c(col_idx, (1:ncol(x_train))[-col_idx])]
x_val <- x_val[, c(col_idx, (1:ncol(x_val))[-col_idx])]

gbm1 <- gbm(y~.,        
            data=x_train,                   
            distribution="bernoulli",     
            n.trees=200,
            n.minobsinnode=20,
            shrinkage=0.03,              
            interaction.depth=8,         
            bag.fraction = 0.5,          
            train.fraction = 0.5,       
            #cv.folds = 3,  
            keep.data=TRUE,              
            verbose=TRUE               
            #n.cores=6
            )      


# check performance using a 50% heldout test set
best.iter <- gbm.perf(gbm1,method="test")
print(best.iter)

### See AUC for validation set
f.predict <- predict(gbm1,x_val,n.trees=best.iter,type="response")
auc(x_val$y, f.predict)


### Predict on test set
t.predict <- predict(gbm1,test,n.trees=best.iter,type="response")


### Write output
sub <- read_csv(paste0(path, "sample_submission.csv", collapse = ""))
sub$target <- t.predict
subversion <- 1
write_csv(sub, paste0(path, "output/test_submission_GBM_", subversion, ".csv", collapse = ""))





##############
#
#   glmnet
#


###x_train_scale <- scale(x_train[,-cat_columns], center = TRUE, scale = TRUE)


gnet <- glmnet(as.matrix(x_train[,2:1461]), x_train$y,
                  family="binomial", alpha=0, standardize=FALSE)

plot(gnet)

g.predict <- predict(gnet, newx = as.matrix(x_val[,2:1461]), type="response")[,1]
auc(x_val$y, g.predict)















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
train <- train[,-c(1, 1934)]

cat_columns <- c()
for (f in 1:ncol(train)) {
  if (class(train[[f]])=="character") {
    cat_columns <- c(cat_columns, f)
  }
}

train_small <- train[1:100,cat_columns]
train_small <- data.frame(lapply(train_small, as.character), stringsAsFactors=FALSE)
#train_small[] <- lapply(train_small, as.character)

freq_matrix <- table( unlist( unname(train_small) ) ) # Other way to count the occurrences
apply( train_small,1, function(x) { paste0( sum(freq_matrix[x]) ,"/", length(x) )}) 

freq_matrix["R"]


train_small[,"Score_text"] <- apply( train_small,1, function(x) { paste0( sum(freq_matrix[x]) ,"/", length(x) )}) 
train_small[,"Score"] <- apply(train_small,1,function(x) { sum(freq_matrix[x]) / length(x) })


