import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-6.781465290499838
exported_pipeline = LGBMRegressor(colsample_bytree=0.8, learning_rate=0.001, max_bin=31, max_depth=4, min_child_weight=25, n_estimators=200, num_leaves=40, objective="mape", reg_alpha=0.0005, subsample=0.9, subsample_freq=30, verbosity=-1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
