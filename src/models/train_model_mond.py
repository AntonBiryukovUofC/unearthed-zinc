from src.models.model import XGBoost, QuantileGB, SVM, QuantileRF,train_model,TimeSeriesSplitImproved,MondrianRF
import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.model_selection import TimeSeriesSplit,ParameterGrid
import pickle

import pandas as pd

import logging
from pathlib import Path
import matplotlib.pyplot as plt


def main(year,tgt):
    import pickle
    import time
    import pandas as pd

    timestr = time.strftime("%Y%m%d-%H%M%S")

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
   # tgt = 'final.output.recovery'
    #tgt = 'rougher.output.recovery'

    #year = 2017
    note = '_1'
    mod = 'mond'
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

    X = data_dict[year]['X_train_pca']
    y = data_dict[year]['y_train']
    print(f'X_train shape: {X.shape}, y_train: {y.shape}')

    X_test = data_dict[year]['X_test_pca']
    #inds = (X['rougher.input.feed_zn'] > 0.5).index
    ### CHECK IF SAMPLING IS ON!
    inds_y = y[(y[tgt] > 5) & (y[tgt] < 100)].index
    inds_common = inds_y

    X = X.loc[inds_common,]
    y = y.loc[inds_common, tgt]

    param_grids = {
                   'min_samples_split':[2,6,10,15,25],
                   'max_depth': [6,8,10,14,16],
                   }
    default = {
                'n_estimators':100,
               'n_jobs': -1,
                'random_state':123
               }

    # n_estimators: Any = 10,
    # max_depth: Any = None,
    # min_samples_split: Any = 2,

    grids = ParameterGrid(param_grids)
    Nmonths_total = 8
    Nspl = int(Nmonths_total * 30 / 10)
    Nmonths_test = 4
    Nmonths_min_train = 2.5
    cv = TimeSeriesSplitImproved(n_splits=Nspl)

    mus = []
    sds = []
    grids_full=[]
    for i in trange(len(grids)):
        g = grids[i]
        g = {**g, **default}
        scores, mu, sd, m = train_model(X, y, cv, model=MondrianRF, params=g,fixed_length=False, train_splits=Nspl // Nmonths_total * Nmonths_min_train, test_splits=int(Nmonths_test / Nmonths_total * Nspl))
        grids_full.append(g)
        mus.append(mu)
        sds.append(sd)

    # Plot the figures!:
    mus = np.array(mus)
    sds = np.array(sds)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.fill_between(np.arange(len(grids)), y1=mus - sds, y2=mus + sds)
    ax.plot(np.arange(len(grids)), mus, '-r')

    labs = [str(g) for g in grids_full]
    ax.set_xticks(np.arange(len(grids_full)))
    ax.set_xticklabels(labs, rotation=90)

    fig.savefig(f'{root}/results/{mod}_{tgt}_{year}_{note}.png')

    id_grid = np.argmin(mus)
    grid_best = grids_full[id_grid]
    print(f'Best score: {mus[id_grid]} +- {sds[id_grid]} at grid = {grid_best}, {tgt} -- {year}')
    m.fit_final(X, y, params=grid_best)
    ypred= m.predict(X_test)
    preds = pd.DataFrame(data = {'date':X_test.index, tgt:ypred})

    preds.to_csv(f'{root}/results/{mod}_{tgt}_{year}_{note}.csv',index=False)
    with open(f'{root}/results/{mod}_{tgt}_{year}_{note}.pkl', 'wb') as f:
        pickle.dump(m.model, f)
if __name__ == '__main__':
    # tgt = 'final.output.recovery'
    # tgt = 'rougher.output.recovery'
    # year = 2016
    for y in [2017]:
        for tgt in ['rougher.output.recovery','final.output.recovery']:
            main(y, tgt)
    # grids[14]
    # Set up crossvalidation procedure:



