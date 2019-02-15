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

# Average CV score on the training set was:-6.110574906923211
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_depth=5, max_features=0.45000000000000007, min_samples_leaf=0.20500000000000002, min_samples_split=0.20500000000000002, n_estimators=500)),
    LGBMRegressor(colsample_bytree=0.65, learning_rate=0.005, max_bin=63, max_depth=7, min_child_weight=1, n_estimators=350, num_leaves=80, objective="fair", reg_alpha=0, subsample=0.55, subsample_freq=20, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)