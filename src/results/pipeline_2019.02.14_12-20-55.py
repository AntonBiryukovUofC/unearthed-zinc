import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-5.8040774392705385
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_depth=4, max_features=0.05, min_samples_leaf=0.055, min_samples_split=0.15500000000000003, n_estimators=400)),
    LGBMRegressor(colsample_bytree=0.8, learning_rate=0.5, max_bin=255, max_depth=5, min_child_weight=15, n_estimators=300, num_leaves=90, objective="huber", reg_alpha=0.07, subsample=0.95, subsample_freq=20, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)