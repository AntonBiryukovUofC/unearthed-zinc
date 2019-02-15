import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler, StandardScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-6.466429790249543
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LGBMRegressor(colsample_bytree=0.75, learning_rate=0.005, max_bin=127, max_depth=5, min_child_weight=5, n_estimators=250, num_leaves=70, objective="mape", reg_alpha=0.01, subsample=0.7, subsample_freq=10, verbosity=-1)),
    StandardScaler(),
    RobustScaler(),
    LGBMRegressor(colsample_bytree=0.8, learning_rate=0.001, max_bin=127, max_depth=4, min_child_weight=2, n_estimators=250, num_leaves=40, objective="mape", reg_alpha=0.05, subsample=0.85, subsample_freq=30, verbosity=-1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)