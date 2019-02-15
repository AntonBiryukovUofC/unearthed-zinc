import pandas as pd
import seaborn as sns
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from ocp_table_tpot.globals import Globals as gd
from tpot import TPOTRegressor
from src.models.model import HistoricalMedian,XGBoost,LinearModel,RF,KNN,SVM,mase,TimeSeriesSplitImproved,my_custom_scorer
from sklearn.metrics import make_scorer

root = Path(__file__).resolve().parents[2]
print(root)

def mase_error(y_true,y_pred):
    return mase(y_pred,y_true)


df_tsfresh = pd.read_pickle(f'{root}/data/processed/train_test_tsfresh.pkl').reset_index(level = 0)
data_dict = pd.read_pickle(f'{root}/data/processed/data_dict_all.pkl')

year = 2016
tgt = 'rougher.output.recovery'
X = data_dict[year]['X_train']
y = data_dict[year]['y_train'][tgt].dropna()


Nmonths_total = 8
Nspl = int(Nmonths_total * 30 / 15)
Nmonths_test = 4
Nmonths_min_train = 2.5
cv = TimeSeriesSplitImproved(n_splits=Nspl)

X = X.sample(frac=0.3).sort_index()
inds = X.index.intersection(y.index)

X = X.loc[inds]
y = y.loc[inds]

gens = 50
pops = 50
seed  = 123

sc= my_custom_scorer()

tpot = TPOTRegressor(generations=50,
                                     verbosity=2,
                                     cv=cv,
                                     scoring=gd.SCORING_LOOKUP_DICT['mae'],
                                     config_dict=gd.CONFIG_LOOKUP_DICT['rf_boost'],
                                     population_size=50,
                                     random_state=seed,
                                     periodic_checkpoint_folder='../results/',
                                     n_jobs=10,
                                     memory=None)
tpot.fit(X,y)
tpot.export(f'../results/tpot-pipeline_{year}_{tgt}.py')