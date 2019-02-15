import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-5.426574112626802
exported_pipeline = ExtraTreesRegressor(bootstrap=False, max_depth=9, max_features=0.35000000000000003, min_samples_leaf=0.005, min_samples_split=0.005, n_estimators=400)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
