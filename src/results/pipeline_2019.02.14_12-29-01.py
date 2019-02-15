import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-5.61990515418617
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_depth=6, max_features=0.5500000000000002, min_samples_leaf=0.055, min_samples_split=0.35500000000000004, n_estimators=100)),
    RobustScaler(),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_depth=8, max_features=0.25000000000000006, min_samples_leaf=0.055, min_samples_split=0.35500000000000004, n_estimators=350)),
    LGBMRegressor(colsample_bytree=0.75, learning_rate=0.5, max_bin=127, max_depth=4, min_child_weight=0.001, n_estimators=100, num_leaves=40, objective="huber", reg_alpha=0.0005, subsample=0.8, subsample_freq=20, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
