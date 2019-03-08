import pickle
import sys

import numpy as np
import pandas as pd
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin

sys.path.insert(0, '..')
from src.models.model import mase
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
def fair_obj(preds, dtrain):
    """y = c * abs(x) - c**2 * np.log(abs(x)/c + 1)"""
    x = preds - dtrain.get_labels()
    c = 1
    den = abs(x) + c
    grad = c*x / den
    hess = c*c / den ** 2
    return grad, hess

def cv_boost(model, X_base=None, y=None):
    train_index = X_base.index.year.isin([2016, 2017])
    valid_index = X_base.index.year.isin([2018])
    X_train, X_valid = X_base.iloc[train_index], X_base.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    model.fit(X_train.values, y_train.values.reshape(-1, ))
    # ntree = model.best_iteration_
    preds = model.predict(X_valid.values)
    oof_scores = mase(preds, y_valid.values)
    return oof_scores


project_dir = Path(__file__).resolve().parents[2]
print(project_dir)
data_dict = pd.read_pickle(f'{project_dir}/data/processed/data_dict_all.pkl')

# Load the data
year = 2019
tgt = 'rougher.output.recovery'
X = data_dict[year][f'X_train_tsclean']  # .drop(cols_drop,axis = 1)
y = data_dict[year]['y_train']
mask = data_dict[year]['mask']
X = X.loc[mask.index, :][mask]
y = y.loc[mask.index, :][mask]
print(f'2)  train: {X.shape}')
y = y.loc[X.index, tgt]
X_filt = X.filter(regex="rougher|dayw|hour", axis=1)
print(f'after sample() train: {X.shape}')
# Load the feature importances

feature_subset_df = pd.read_csv(f'{project_dir}/notebooks/shap-importance-rougher-diff_deriv.csv')
X_sub = X_filt.copy()
X_sub.columns = [x.replace("\"", "") for x in X_sub.columns]
# Actual model training:
n_array = np.arange(start=50, stop=feature_subset_df.shape[0], step=50, dtype='int')
for N in n_array:
    print(f'Evaluating {N} features !')
    feature_subset = feature_subset_df['feature'].head(N)
    X_sub_check = X_sub[feature_subset]
    trials = Trials()

    fpath = f'{project_dir}/models/{N}_feats_rougher_Trials_xgb.pkl'
    fpath_csv = f'{project_dir}/models/{N}_feats_rougher_Trials_xgb.csv'
    def objective(params):
        params = {

            'max_depth': int(params['max_depth']),
            'gamma': "{:.3f}".format(params['gamma']),
            'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
            'subsample': "{:.3f}".format(params['subsample']),
            'min_child_weight': "{:.3f}".format(params['min_child_weight']),
            'alpha': params['alpha'],
            'lambda': params['lambda']}

        m = xgb.XGBRegressor(learning_rate=0.05, obj = fair_obj,
                             n_estimators=900,
                             random_state =123,verbose=0,silent=True, n_jobs = -1,**params)
        print(X_sub_check.shape)
        sc = cv_boost(m, X_base=X_sub_check, y=y)
        print("Score {:.3f} params {}".format(sc, params))
        return sc


    space = {
        'max_depth': hp.quniform('max_depth', 2, 4, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.05, 1.0),
        'min_child_weight': hp.loguniform('min_child_weight', -4, 5),
        'subsample': hp.uniform('subsample', 0.3, 1.0),
        'gamma': hp.uniform('gamma', 1, 100.),
        'alpha': hp.uniform('alpha', 1, 50.),
        'lambda': hp.uniform('lambda', 0, 50.)

    }
    best_xgb = fmin(fn=objective,
                     space=space,
                     algo=tpe.suggest,
                     max_evals=100,trials = trials)
    losses = [trials.trials[i]['result']['loss'] for i in range(len(trials.trials))]
    params = pd.DataFrame(trials.vals)
    params['loss'] = losses
    params.sort_values('loss', inplace=True)
    params.to_csv(fpath_csv)
    with open(fpath, 'wb') as f:
        pickle.dump(trials, f)