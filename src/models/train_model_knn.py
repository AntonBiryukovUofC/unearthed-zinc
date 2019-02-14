from src.models.model import XGBoost, QuantileGB, SVM, QuantileRF,train_model
import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.model_selection import TimeSeriesSplit,ParameterGrid
import pickle

import pandas as pd

import logging
from pathlib import Path



if __name__ == '__main__':
    import pickle
    import time
    import pandas as pd

    timestr = time.strftime("%Y%m%d-%H%M%S")

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    tgt = 'final.output.recovery'
    #tgt = 'rougher.output.recovery'

    year = 2017
    note = '_1'
    # not used in this stub but often useful for finding various file
    root = Path(__file__).resolve().parents[2]
    print(root)
    # Get raw features
    with open(f'{root}/data/processed/data_dict_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    # X_train is used for training and validation, X_test - final predictions (we have no labels for it)
    # Fix the year at 2016 for now
    #X_train = data_dict[year]['X_train']
    #y_train = data_dict[year]['y_train']

    X = data_dict[year]['X_train_ts']
    y = data_dict[year]['y_train']
    print(f'X_train shape: {X.shape}, y_train: {y.shape}')

    X_test = data_dict[year]['X_test_ts']
    inds = (X['rougher.input.feed_zn'] > 0.5).index
    inds_y = y[y[tgt] > 0].index
    inds_common = inds_y.intersection(inds)

    X = X.loc[inds_common,]
    y = y.loc[inds_common, tgt]

    param_grids = {'n_estimators': [50],
                   'min_samples_leaf':[2,5,10,20],
                   'max_features': [0.05,0.1,0.3,0.5],
                   'max_depth': [4,6,10,14,18],
                   }
    default = {
               'criterion': 'mae',
               'n_jobs': -1
               }

    # n_estimators: Any = 10,
    # criterion: Any = 'mse',
    # max_depth: Any = None,
    # min_samples_split: Any = 2,
    # min_samples_leaf: Any = 1,
    # min_weight_fraction_leaf: Any = 0.0,
    # max_features: Any = 'auto',
    # max_leaf_nodes: Any = None,
    # bootstrap: Any = True,
    # oob_score: Any = False,
    # n_jobs: Any = 1,
    # random_state: Any = None,
    # verbose: Any = 0,
    # warm_start: Any = False) -> None

    grids = ParameterGrid(param_grids)
    cv = TimeSeriesSplit(n_splits=5)

    mus = []
    sds = []
    grids_full=[]
    for i in trange(len(grids)):
        g = grids[i]
        g = {**g, **default}
        scores, mu, sd, m = train_model(X, y, cv, model=QuantileRF, params=g)
        grids_full.append(g)
        mus.append(mu)
        sds.append(sd)

    id_grid = np.argmin(mus)
    grid_best = grids_full[id_grid]
    print(f'Best score: {mus[id_grid]} +- {sds[id_grid]} at grid = {grid_best}')
    m.fit_final(X, y, params=grid_best)
    ypred= m.predict(X_test)
    preds = pd.DataFrame(data = {'date':X_test.index, tgt:ypred})

    preds.to_csv(f'{root}/results/qrf_{tgt}_{year}_{note}.csv',index=False)



    # grids[14]
    # Set up crossvalidation procedure:



