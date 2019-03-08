import pandas as pd
import pickle
from pathlib import Path
import seaborn as sns
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import os
import sys
from ocp_table_tpot.globals import Globals as gd
from tpot import TPOTRegressor
sys.path.insert(0,'..')

from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler,MinMaxScaler,PolynomialFeatures,QuantileTransformer,Normalizer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error,make_scorer
from copy import copy
from tpot.builtins import StackingEstimator


from src.models.model import mase,TimeSeriesSplitImproved
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,RANSACRegressor,Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from skgarden.quantile import RandomForestQuantileRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
import lightgbm as lgb
import umap


# This function takes one model and fit it to the train and test data
# It returns the model MASE, CV prediction, and test prediction
def base_fit(model, folds, features, target, trainData, testData):
    # Initialize empty lists and matrix to store data
    model_mase = []
    model_val_predictions = np.empty((trainData.shape[0], 1))
    k = 0
    # Loop through the index in KFolds
    model_test_predictions = np.zeros((testData.shape[0],))
    model_val_true = np.zeros((trainData.shape[0], 1))

    for train_index, val_index in folds.split(trainData):
        k = k + 1
        # Split the train data into train and validation data
        train, validation = trainData.iloc[train_index], trainData.iloc[val_index]
        # Get the features and target
        train_features, train_target = train[features], train[target]
        validation_features, validation_target = validation[features], validation[target]

        # Fit the base model to the train data and make prediciton for validation data
        if (model.__class__ == xgb.sklearn.XGBRegressor) | (model.__class__ == lgb.sklearn.LGBMRegressor):
         #   print('Fitting a boost model with limited tree rounds')
            evalset = [(validation_features, np.ravel(validation_target))]
            model.fit(train_features, np.ravel(train_target), eval_set=evalset, verbose=False)
        else:
            model.fit(train_features, train_target.values)

        if (model.__class__ == xgb.sklearn.XGBRegressor):
         #   print(model.best_ntree_limit)
         #   print('Using xgboost with limited tree rounds')
            validation_predictions = model.predict(validation_features, ntree_limit=model.best_ntree_limit)

        elif (model.__class__ == lgb.sklearn.LGBMRegressor):
         #   print(model.best_iteration_)
        #   print('Using lgbmboost with limited tree rounds')
            validation_predictions = model.predict(validation_features, num_iteration=model.best_iteration_)
        else:
            print('Using generic predict')
            validation_predictions = model.predict(validation_features)

        # Calculate and store the MASE for validation data
       # print(mase(validation_predictions, validation_target))
        # model_mase.append(mase(validation_predictions,validation_target))

        # Save the validation prediction for level 1 model training
        model_val_predictions[val_index, 0] = validation_predictions.reshape(validation.shape[0])
        model_val_true[val_index, 0] = validation_target.values
        model_test_predictions += model.predict(testData[features])

    model_test_predictions = model_test_predictions / k
    # Fit the base model to the whole training data
    # model.fit(trainData[features], np.ravel(trainData[target]))
    # Get base model prediction for the test data
    # model_test_predictions = model.predict(testData[features])
    # Calculate and store the MASE for validation data

    # model_val_predictions = model_val_predictions
    model_mase.append(mase(model_val_predictions, model_val_true))

    return model_mase, model_val_predictions, model_test_predictions


# Function that takes a dictionary of models and fits it to the data using baseFit
# The results of the models are then aggregated and returned for level 1 model training
def stacks(level0_models, folds, features, target, trainData, testData):
    num_models = len(level0_models.keys())  # Number of models

    # Initialize empty lists and matrix
    level0_trainFeatures = np.empty((trainData.shape[0], num_models))
    level0_testFeatures = np.empty((testData.shape[0], num_models))

    # Loop through the models
    for i, key in enumerate(level0_models.keys()):
        print('Fitting %s -----------------------' % (key))
        model_mase, val_predictions, test_predictions = base_fit(level0_models[key], folds, features, target, trainData,
                                                                 testData)

        # Print the average MASE for the model
        print('%s average MASE: %s' % (key, np.mean(model_mase)))
        print('\n')

        # Aggregate the base model validation and test data predictions
        level0_trainFeatures[:, i] = val_predictions.reshape(trainData.shape[0])
        level0_testFeatures[:, i] = test_predictions.reshape(testData.shape[0])

    return level0_trainFeatures, level0_testFeatures


def stackerTraining(stacker, folds, level0_trainFeatures, trainData, target=None):
    for k in stacker.keys():
        print('Training stacker %s' % (k))
        stacker_model = stacker[k]
        stacker_mase = []
        y_pred = np.zeros_like(trainData[target].values)
        y_true = np.zeros_like(trainData[target].values)
        for t, v in folds.split(X, y):
            train, validation = level0_trainFeatures[t, :], level0_trainFeatures[v, :]
            # Get the features and target
            train_features, train_target = train, trainData.iloc[t][target]
            validation_features, validation_target = validation, trainData.iloc[v][target]

            if (stacker_model.__class__ == xgb.sklearn.XGBRegressor) | (
                    stacker_model.__class__ == lgb.sklearn.LGBMRegressor):
                print('Fitting a boost model with limited tree rounds')
                evalset = [(validation_features, np.ravel(validation_target))]
                stacker_model.fit(train_features, np.ravel(train_target), eval_set=evalset, early_stopping_rounds=20,
                                  verbose=False)
                print(stacker_model.best_iteration_)
            else:
                stacker_model.fit(level0_trainFeatures[t, :], train_target)

            y_pred[v] = stacker_model.predict(level0_trainFeatures[v])
            y_true[v] = trainData.iloc[v][target].values

        stacker_mase = mase(y_pred, y_true)
        average_mase = mase(level0_trainFeatures.mean(axis=1), y_true)
        print('%s Stacker MASE: %s' % (k, stacker_mase))
        print('%s Averaging MASE: %s' % (k, average_mase))


project_dir = Path(__file__).resolve().parents[2]
print(project_dir)
data_dict = pd.read_pickle(f'{project_dir}/data/processed/data_dict_all.pkl')


### Load the data, deal with the outliers, print the shapes!

feature_subset_rougher_df = pd.read_csv(f'{project_dir}/notebooks/shap-importance-rougher-diff_deriv_normalized_with_interaction.csv')
feature_subset_final_df = pd.read_csv(f'{project_dir}/notebooks/shap-importance-final-diff_deriv_normalized_with_interaction.csv')


year = 2019
tgt = 'rougher.output.recovery'
X = data_dict[year]['X_train_tsclean']
X.columns = [x.replace("\"","") for x in X.columns]
print(X.shape)
y = data_dict[year]['y_train']
X_test = data_dict[year]['X_test_ts']
mask = data_dict[year]['mask']
exclude_pts = data_dict[year]['excl'].set_index('date').tz_localize('UTC')
inds = mask.index.difference(exclude_pts.index)
X=X.loc[inds,:]
y=y.loc[inds,:]
mask=mask[inds]
print(X.shape)
print(X_test.shape)


# Subset the points by mask and target
X = X[mask]
y = y[mask][tgt]
# Filter to upstream variables
print(f'2) Train shape: {X.shape}')
X_filt = X.filter(regex  ="rougher|hour|dayw",axis = 1)
X = X_filt
train_df = pd.concat([X,y],axis= 1)

# Get the K fold indexes
n_folds = 15
kf = KFold(n_splits=n_folds, shuffle=False, random_state=156)
N_lvl_dicts = {}
# Gather models first:
n_array = np.arange(start=50, stop=550, step=100, dtype='int')
for N in n_array:
    level0_models = {}
    fpath_csv = f'{project_dir}/models/{N}_feats_rougher_Trials.csv'

    res_df = pd.read_csv(fpath_csv)
    res_df['num_leaves'] = res_df['num_leaves'].astype('int')
    for i in range(6):
        params = res_df[['bagging_fraction','feature_fraction','lambda_l1','num_leaves']].iloc[i].to_dict()
        params['num_leaves'] = int(params['num_leaves'])
        level0_models[f'lgbm_{i}_{N}'] = lgb.LGBMRegressor(objective='mae',n_jobs=8,verbose = -1,
                              learning_rate=0.05, n_estimators=900,
                              **params)
    N_lvl_dicts[N] = level0_models

# Loop over the new dict now, and predict on test / train:
N_testfeatures = {}
N_trainfeatures = {}
# Now train them all on an appropriate dataset:
for N in n_array:
    lvl_mod = N_lvl_dicts[N]
    feature_subset = feature_subset_rougher_df['feature'].head(N)
    level0_trainFeatures_rougher, level0_testFeatures_rougher = stacks(lvl_mod, kf, feature_subset,
                                                                       tgt, train_df, X_test)
    N_testfeatures[N] = level0_testFeatures_rougher
    N_trainfeatures[N] = level0_trainFeatures_rougher
fpath_test = f'{project_dir}/models/test_features_rougher.pkl'
fpath_train = f'{project_dir}/models/train_features_rougher.pkl'

# Pickle the features
with open(fpath_test,'wb') as f:
    pickle.dump(N_testfeatures,f)

with open(fpath_train,'wb') as f:
    pickle.dump(N_trainfeatures,f)


