import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-5.218540992702212
exported_pipeline = make_pipeline(
    RobustScaler(),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_depth=8, max_features=0.8500000000000002, min_samples_leaf=0.255, min_samples_split=0.255, n_estimators=250)),
    MinMaxScaler(),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_depth=9, max_features=0.6500000000000001, min_samples_leaf=0.005, min_samples_split=0.10500000000000001, n_estimators=400)),
    LGBMRegressor(colsample_bytree=0.7, learning_rate=0.1, max_bin=127, max_depth=6, min_child_weight=3, n_estimators=300, num_leaves=70, objective="huber", reg_alpha=0.03, subsample=0.65, subsample_freq=5, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
