import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-5.128385215987203
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_depth=5, max_features=0.7500000000000002, min_samples_leaf=0.15500000000000003, min_samples_split=0.20500000000000002, n_estimators=350)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_depth=8, max_features=0.5500000000000002, min_samples_leaf=0.055, min_samples_split=0.10500000000000001, n_estimators=500)),
    LGBMRegressor(colsample_bytree=0.55, learning_rate=0.1, max_bin=63, max_depth=6, min_child_weight=10, n_estimators=300, num_leaves=250, objective="huber", reg_alpha=0.1, subsample=0.5, subsample_freq=20, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
