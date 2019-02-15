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

# Average CV score on the training set was:-6.072446819385328
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_depth=4, max_features=0.9500000000000002, min_samples_leaf=0.10500000000000001, min_samples_split=0.005, n_estimators=150)),
    StackingEstimator(estimator=LGBMRegressor(colsample_bytree=0.7, learning_rate=0.05, max_bin=63, max_depth=4, min_child_weight=25, n_estimators=350, num_leaves=150, objective="huber", reg_alpha=0.003, subsample=0.9, subsample_freq=10, verbosity=-1)),
    LGBMRegressor(colsample_bytree=0.75, learning_rate=0.01, max_bin=63, max_depth=7, min_child_weight=15, n_estimators=300, num_leaves=90, objective="fair", reg_alpha=0.05, subsample=0.75, subsample_freq=10, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
