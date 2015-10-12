Kaggle Springleaf competition 
=====================

The Springlead competition page can be found at https://www.kaggle.com/c/springleaf-marketing-response

# Generating the solution:

## Data Deep Dive
A look at the data is given in `Explore_Data.R`


## Generating Training and Testing datasets
Run `Preprocess.R` to generate new training and testing sets. 
This will create the datasets for NearZeroVariance removed, correlations removed, and an interaction dataset.

## XGBoost Models
Train multiple XGBoost models with `XGBoost_Step1.R`.
We can try multiple scenarios of the base data, removing NearZeroVariance, adding interactions, etc.
We can start by splitting the training set into 70/30 train/validation and determining the value
of nrounds where validation stops improving. Next, train on full dataset and same parameters with
that found value of nrounds. 

## H2o Models
Train different H2o models in `H2o.R`.

## Train online models
We can train FTLR and SGD online models with `pypy FTLR.py` and `pypy SGD.py`.
As a helper, the scripts `pypy run_FTLR.py` and `pypy run_SGD.py` will run multiple models. 

## Average Ensemble
A simple improvement can be achieved by averaging all models together. 
Run `python kaggle_avg.py "sub*.csv" "kaggle_avg.csv"` 
Also, within the notebook `xgboost.ipynb` we can do a weighted average.


## Model Stacking (Two Stage Model)
As a way to improve the final performance perform stacking with CV. 
We can take all the XGBoost, H2o, and online models and create a two stage model.
The script to perform model stacking is `Stacking.R`.

## Bagging
A final improvement on the two stage model is to perform bagging on the 2nd level model.
This script is found in `Bagging.R`.