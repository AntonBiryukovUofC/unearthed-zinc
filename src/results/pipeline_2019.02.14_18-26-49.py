import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-5.458658841136811
exported_pipeline = LGBMRegressor(colsample_bytree=0.7, learning_rate=0.1, max_bin=127, max_depth=5, min_child_weight=3, n_estimators=300, num_leaves=70, objective="huber", reg_alpha=0.03, subsample=0.65, subsample_freq=5, verbosity=-1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
