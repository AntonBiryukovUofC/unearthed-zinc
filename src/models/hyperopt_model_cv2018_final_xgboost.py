import pickle
import sys

import numpy as np
import pandas as pd
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin

sys.path.insert(0, '..')
from src.models.model import mase
import lightgbm as lgb
from pathlib import Path


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
tgt = 'final.output.recovery'
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

feature_subset_df = pd.read_csv(f'{project_dir}/notebooks/shap-importance-final-diff_deriv.csv')
X_sub = X.copy()
X_sub.columns = [x.replace("\"", "") for x in X_sub.columns]
# Actual model training:
n_array = np.arange(start=50, stop=feature_subset_df.shape[0], step=100, dtype='int')
for N in n_array:
    print(f'Evaluating {N} features !')
    feature_subset = feature_subset_df['feature'].head(N)
    X_sub_check = X_sub[feature_subset]
    trials = Trials()

    fpath = f'{project_dir}/models/{N}_feats_final_Trials_xgb.pkl'
    fpath_csv = f'{project_dir}/models/{N}_feats_final_Trials_xgb.csv'

    def objective(params):
        params = {
            # 'max_depth': int(params['max_depth']),
            'num_leaves': int(params['num_leaves']),  # int(max(2**(int(params['max_depth'])) - params['num_leaves'],0)),
            'feature_fraction': "{:.3f}".format(params['feature_fraction']),
            'bagging_fraction': '{:.3f}'.format(params['bagging_fraction']),
            'lambda_l1': params['lambda_l1']
            # "min_data_in_leaf": int(params['min_data_in_leaf'])
        }
        m = lgb.LGBMRegressor(objective='mae',n_jobs=8,
                              learning_rate=0.05, n_estimators=900, random_state=9,
                              **params)
        print(X_sub_check.shape)
        sc = cv_boost(m, X_base=X_sub_check, y=y)
        print("Score {:.3f} params {}".format(sc, params))
        return sc


    space = {
        # 'max_depth': hp.quniform('max_depth', 2, 8, 1),
        'num_leaves': hp.quniform('num_leaves', 8, 50, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.02, 0.9),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.1, 1.0),
        'lambda_l1': hp.uniform('lambda_l1', 0.1, 80)
        # "min_data_in_leaf": hp.loguniform("min_data_in_leaf",1,8)
    }
    best_lgbm = fmin(fn=objective,
                     space=space,
                     algo=tpe.suggest,
                     max_evals=100,trials = trials)
    losses = [trials.trials[i]['result']['loss'] for i in range(len(trials.trials))]
    params = pd.DataFrame(trials.vals)
    params['loss'] = losses
    params.sort_values('loss',inplace= True)
    params.to_csv(fpath_csv)
    with open(fpath,'wb') as f:
        pickle.dump(trials,f)