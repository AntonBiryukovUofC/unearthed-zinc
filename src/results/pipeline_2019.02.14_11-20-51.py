import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-5.797339389160764
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_depth=5, max_features=0.15000000000000002, min_samples_leaf=0.055, min_samples_split=0.005, n_estimators=100)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_depth=9, max_features=0.15000000000000002, min_samples_leaf=0.255, min_samples_split=0.055, n_estimators=100)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_depth=5, max_features=0.35000000000000003, min_samples_leaf=0.20500000000000002, min_samples_split=0.35500000000000004, n_estimators=500)),
    LGBMRegressor(colsample_bytree=0.75, learning_rate=0.01, max_bin=127, max_depth=4, min_child_weight=15, n_estimators=300, num_leaves=90, objective="fair", reg_alpha=0.05, subsample=0.75, subsample_freq=0, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
