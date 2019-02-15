import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-5.030601467219057
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_depth=4, max_features=0.9500000000000002, min_samples_leaf=0.15500000000000003, min_samples_split=0.005, n_estimators=150)),
    StackingEstimator(estimator=XGBRegressor(colsample_bylevel=0.85, colsample_bytree=0.85, eval_metric="mae", learning_rate=0.001, max_depth=5, min_child_weight=0.001, n_estimators=90, n_jobs=1, reg_alpha=0, subsample=0.5)),
    MinMaxScaler(),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_depth=6, max_features=0.7500000000000002, min_samples_leaf=0.20500000000000002, min_samples_split=0.255, n_estimators=350)),
    ExtraTreesRegressor(bootstrap=False, max_depth=9, max_features=0.25000000000000006, min_samples_leaf=0.005, min_samples_split=0.15500000000000003, n_estimators=400)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
