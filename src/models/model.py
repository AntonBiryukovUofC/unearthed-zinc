import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

np.random.seed(123)
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import make_scorer

DROPCOLS = ['primary_cleaner.state.floatbank8_a_level',
            'rougher.state.floatbank10_a_level',
            'secondary_cleaner.state.floatbank6_a_level',
            'rougher.state.floatbank10_f_level',
            'rougher.state.floatbank10_e_level',
            'secondary_cleaner.state.floatbank4_a_level',
            'rougher.state.floatbank10_c_level',
            'rougher.state.floatbank10_d_level',
            'secondary_cleaner.state.floatbank4_a_air',
            'rougher.state.floatbank10_c_level',
            'secondary_cleaner.state.floatbank3_b_level',
            'secondary_cleaner.state.floatbank4_b_level',
            'secondary_cleaner.state.floatbank2_b_level',
            ]
DROPCOLS = ['rougher.input.floatbank10_copper_sulfate',
            'rougher.input.floatbank10_xanthate',
            'rougher.state.floatbank10_b_air',
            'rougher.state.floatbank10_e_air',
            'rougher.state.floatbank10_f_air',
            'primary_cleaner.state.floatbank8_b_air',
            'primary_cleaner.state.floatbank8_c_air',
            "secondary_cleaner.state.floatbank4_b_air",
            'secondary_cleaner.state.floatbank2_b_air',
            "secondary_cleaner.state.floatbank5_b_air",
            "secondary_cleaner.state.floatbank3_a_air"
            ]




def _mase_numeric_only(predicted, measured):
    naive_forecast_error = np.abs(measured[1:] - measured[:-1]).mean()
    forecast_error = \
        np.abs(measured - np.nan_to_num(predicted)) / naive_forecast_error
    return np.nanmean(forecast_error)


def mase(predicted, measured, min_samples=3):
    if min_samples < 2:
        raise ValueError('mase.min_samples must be at least 2')

    # Make sure we have numpy arrays
    predicted = np.asarray(predicted)
    measured = np.asarray(measured)

    # Apply MASE over all the non-NaN slices with at least 3 hours of data
    if np.isnan(measured).any():
        segments = [
            _mase_numeric_only(predicted[_slice], measured[_slice])
            for _slice in np.ma.clump_unmasked(np.ma.masked_invalid(measured))
            if abs(_slice.stop - _slice.start) > min_samples
        ]
        if not segments:
            raise ValueError("Couldn't find any non-NaN segments longer than "
                             "{} in measurements".format(min_samples))
        score = np.mean(segments)
    else:
        if len(measured) < min_samples:
            raise ValueError('Need at least {} samples to calculate MASE'.format(min_samples))
        score = _mase_numeric_only(predicted, measured)

    return score



def my_custom_scorer():
    return make_scorer(mase, greater_is_better=False)


def train_model(X, y, folds, model=None, score=mase, params=None, fixed_length=False, train_splits=None,
                test_splits=None):
    import pandas as pd
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(
            folds.split(X, train_splits=train_splits, test_splits=test_splits)):
        # print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        m = model(X_train, y_train, X_valid, y_valid, params=params)
        score_val = m.evaluate(X_val=X_valid, y_val=y_valid)
        # print(f'Fold {fold_n}. Score: {score_val:.4f}.')
        print('')
        scores.append(score_val)
    print(f'CV mean score: {np.mean(scores):.4f}, std: {np.std(scores):.4f}.')
    mu = np.mean(scores)
    sd = np.std(scores)
    return scores, mu, sd, m


#



class TimeSeriesSplitImproved(TimeSeriesSplit):

    def split(self, X, y=None, groups=None, fixed_length=False,
              train_splits=5, test_splits=6):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        fixed_length : bool, hether training sets should always have
            common length
        train_splits : positive int, for the minimum number of
            splits to include in training sets
        test_splits : positive int, for the number of splits to
            include in the test set
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
         Example:
         Nspl = int(Nmonths_total * 30 / 5)
        Nmonths_total = 8
        Nmonths_test = 4
        Nmonths_min_train = 2
        cv_ts = TimeSeriesSplitImproved(n_splits=Nspl)

        k = 0
        tt = pd.DataFrame()
        fig, ax = plt.subplots(figsize=(10, 18))
        for train_index, test_index in cv_ts.split(X, fixed_length=False,
                                           train_splits=Nspl // Nmonths_total * Nmonths_min_train,
                                               test_splits=int(Nmonths_test / Nmonths_total * Nspl)):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(y_test.head(1))
        k += 1
        ax.plot(y_train.index, (y_train * 0 + k))
        ax.plot(y_test.index, (y_test * 0 + k + 0.2))


        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        train_splits, test_splits = int(train_splits), int(test_splits)
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater than the number of samples: {1}.").format(n_folds,
                                                                                                     n_samples))

        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)
        if fixed_length:
            for i, test_start in zip(range(len(test_starts)),
                                     test_starts):
                rem = 0
                if i == 0:
                    rem = n_samples % n_folds
                yield (indices[(test_start - train_size - rem):test_start],
                       indices[test_start:test_start + test_size])
        else:
            for test_start in test_starts:
                yield (indices[:test_start],
                       indices[test_start:test_start + test_size])


# This function takes one model and fit it to the train and test data
# It returns the model MASE, CV prediction, and test prediction
# Create a function to fit a base model on K-1 folds, predict on 1 fold
def base_fit(model, folds, features, target, trainData, testData):
    # Initialize empty lists and matrix to store data
    model_mase = []
    model_val_predictions = np.empty((trainData.shape[0], 1))
    k = 0
    # Loop through the index in KFolds
    model_test_predictions = np.zeros((testData.shape[0],))
    model_val_true = np.zeros((trainData.shape[0], 1))

    for train_index, val_index in folds.split(trainData):
        k = k + 1
        # Split the train data into train and validation data
        train, validation = trainData.iloc[train_index], trainData.iloc[val_index]
        # Get the features and target
        train_features, train_target = train[features], train[target]
        validation_features, validation_target = validation[features], validation[target]

        # Fit the base model to the train data and make prediciton for validation data
        if (model.__class__ == xgb.sklearn.XGBRegressor) | (model.__class__ == lgb.sklearn.LGBMRegressor):
            print('Fitting a boost model with limited tree rounds')
            evalset = [(validation_features, np.ravel(validation_target))]
            model.fit(train_features, np.ravel(train_target), eval_set=evalset, early_stopping_rounds=20, verbose=False)
        else:
            model.fit(train_features, train_target.values)

        if (model.__class__ == xgb.sklearn.XGBRegressor):
            print(model.best_ntree_limit)
            print('Using xgboost with limited tree rounds')
            validation_predictions = model.predict(validation_features, ntree_limit=model.best_ntree_limit)

        elif (model.__class__ == lgb.sklearn.LGBMRegressor):
            print(model.best_iteration_)
            print('Using lgbmboost with limited tree rounds')
            validation_predictions = model.predict(validation_features, num_iteration=model.best_iteration_)
        else:
            print('Using generic predict')
            validation_predictions = model.predict(validation_features)

        # Calculate and store the MASE for validation data
        print(mase(validation_predictions, validation_target))
        # model_mase.append(mase(validation_predictions,validation_target))

        # Save the validation prediction for level 1 model training
        model_val_predictions[val_index, 0] = validation_predictions.reshape(validation.shape[0])
        model_val_true[val_index, 0] = validation_target.values
        model_test_predictions += model.predict(testData[features])

    model_test_predictions = model_test_predictions / k
    # Fit the base model to the whole training data
    # model.fit(trainData[features], np.ravel(trainData[target]))
    # Get base model prediction for the test data
    # model_test_predictions = model.predict(testData[features])
    # Calculate and store the MASE for validation data

    # model_val_predictions = model_val_predictions
    model_mase.append(mase(model_val_predictions, model_val_true))


# Create a function to fit a dictionary of models, and get their OOF predictions from the training data
# Function that takes a dictionary of models and fits it to the data using baseFit
# The results of the models are then aggregated and returned for level 1 model training
def stacks(level0_models, folds, features, target, trainData, testData):
    num_models = len(level0_models.keys())  # Number of models

    # Initialize empty lists and matrix
    level0_trainFeatures = np.empty((trainData.shape[0], num_models))
    level0_testFeatures = np.empty((testData.shape[0], num_models))

    # Loop through the models
    for i, key in enumerate(level0_models.keys()):
        print('Fitting %s -----------------------' % (key))
        model_mase, val_predictions, test_predictions = base_fit(level0_models[key], folds, features, target, trainData,
                                                                 testData)

        # Print the average MASE for the model
        print('%s average MASE: %s' % (key, np.mean(model_mase)))
        print('\n')

        # Aggregate the base model validation and test data predictions
        level0_trainFeatures[:, i] = val_predictions.reshape(trainData.shape[0])
        level0_testFeatures[:, i] = test_predictions.reshape(testData.shape[0])

    return (level0_trainFeatures, level0_testFeatures)


# Function that takes a dictionary of classifiers and train them on base model predictions
def stackerTraining(stacker, folds, level0_trainFeatures, trainData, target=None):
    for k in stacker.keys():
        print('Training stacker %s' % (k))
        stacker_model = stacker[k]
        y_pred = np.zeros_like(trainData[target].values)
        y_true = np.zeros_like(trainData[target].values)
        for t, v in folds.split(trainData, trainData[target]):
            train, validation = level0_trainFeatures[t, :], level0_trainFeatures[v, :]
            # Get the features and target
            train_features, train_target = train, trainData.iloc[t][target]
            validation_features, validation_target = validation, trainData.iloc[v][target]

            if (stacker_model.__class__ == xgb.sklearn.XGBRegressor) | (
                    stacker_model.__class__ == lgb.sklearn.LGBMRegressor):
                print('Fitting a boost model with limited tree rounds')
                evalset = [(validation_features, np.ravel(validation_target))]
                stacker_model.fit(train_features, np.ravel(train_target), eval_set=evalset, early_stopping_rounds=20,
                                  verbose=False)
                print(stacker_model.best_iteration_)
            else:
                stacker_model.fit(level0_trainFeatures[t, :], train_target)

            y_pred[v] = stacker_model.predict(level0_trainFeatures[v])
            y_true[v] = trainData.iloc[v][target].values

        stacker_mase = mase(y_pred, y_true)
        average_mase = mase(level0_trainFeatures.mean(axis=1), y_true)
        print('%s Stacker MASE: %s' % (k, stacker_mase))
        print('%s Averaging MASE: %s' % (k, average_mase))


DROPCOLS_DIFF_FINAL = [
    "diff_week",
    "diff_encod_rel_primary_cleaner.input.copper_sulfate",
    "diff_dayw",
    "diff_encod_rel_primary_cleaner.input.depressant",
    "diff_encod_rel_rougher.input.feed_pb",
    "diff_encod_dif_primary_cleaner.input.depressant",
    "diff_encod_val_primary_cleaner.input.feed_size",
    "diff_encod_rel_primary_cleaner.input.xanthate",
    "diff_encod_dif_primary_cleaner.input.xanthate",
    "diff_daily_avg_final",
    "diff_encod_dif_primary_cleaner.input.feed_size",
    "diff_encod_rel_primary_cleaner.state.floatbank8_a_level",
    "diff_encod_dif_primary_cleaner.state.floatbank8_a_level",
    "diff_hour",
    "diff_daily_avg_rougher",
    "diff_rougher.state.floatbank10_b_level",
    "diff_primary_cleaner.input.feed_size",
    "diff_primary_cleaner.state.floatbank8_a_air",
    "diff_primary_cleaner.state.floatbank8_a_level",
    "diff_primary_cleaner.state.floatbank8_d_air",
    "diff_rougher.input.feed_fe",
    "diff_rougher.input.floatbank11_copper_sulfate",
    "diff_rougher.state.floatbank10_a_air",
    "diff_rougher.state.floatbank10_a_level",
    "diff_rougher.state.floatbank10_c_level",
    "diff_secondary_cleaner.state.floatbank6_a_level",
    "diff_rougher.state.floatbank10_d_air",
    "diff_rougher.state.floatbank10_d_level",
    "diff_secondary_cleaner.state.floatbank2_a_air",
    "diff_secondary_cleaner.state.floatbank2_b_air",
    "diff_secondary_cleaner.state.floatbank2_b_level",
    "diff_secondary_cleaner.state.floatbank3_b_level",
    "diff_secondary_cleaner.state.floatbank4_a_level",
    "diff_secondary_cleaner.state.floatbank5_a_air",
    "diff_secondary_cleaner.state.floatbank5_b_air",
    "diff_secondary_cleaner.state.floatbank5_b_level",
    "diff_rougher.state.floatbank10_e_level",
    "diff_encod_val_primary_cleaner.input.copper_sulfate",

]

DROPCOLS_FINAL = [
    "primary_cleaner.state.floatbank8_b_level",
    "encod_rel_primary_cleaner.input.depressant",
    "secondary_cleaner.state.floatbank2_a_level",
    "rougher.state.floatbank10_e_level",
    "rougher.state.floatbank10_d_level",
    "encod_dif_primary_cleaner.input.depressant",
    "secondary_cleaner.state.floatbank3_a_level",
    "primary_cleaner.state.floatbank8_a_level",
    "hour",
    "secondary_cleaner.state.floatbank4_a_level",
    "secondary_cleaner.state.floatbank3_b_level",
    "secondary_cleaner.state.floatbank5_a_level",
    "encod_val_rougher.input.feed_zn",
    "encod_rel_primary_cleaner.state.floatbank8_a_level",
    "secondary_cleaner.state.floatbank2_b_level",
    "encod_val_primary_cleaner.input.feed_size",
    "secondary_cleaner.state.floatbank6_a_level",
    "rougher.state.floatbank10_a_level",
    "encod_dif_primary_cleaner.state.floatbank8_a_level",
    "secondary_cleaner.state.floatbank3_a_level"
]

DROPCOLS_DIFF_ROUGHER = [
    "diff_rougher.state.floatbank10_c_air",
    "diff_rougher.state.floatbank10_b_air",
    "diff_encod_dif_rougher.input.feed_zn",
    "diff_rougher.input.feed_size",
    "diff_rougher.state.floatbank10_f_air",
    "diff_encod_val_rougher.input.feed_fe",
    "diff_rougher.input.feed_rate",
    "diff_encod_rel_rougher.input.feed_fe",
    "diff_rougher.state.floatbank10_d_level",
    "diff_encod_dif_rougher.input.feed_fe",
    "diff_rougher.state.floatbank10_a_air",
    "diff_encod_dif_rougher.input.feed_pb",
    "diff_encod_rel_rougher.input.feed_pb",
    "diff_rougher.state.floatbank10_b_level",
    "diff_rougher.state.floatbank10_a_level",
    "diff_hour",
    "diff_daily_avg_rougher",
    "diff_rougher.state.floatbank10_f_level",
    "diff_rougher.state.floatbank10_e_level",
    "diff_rougher.state.floatbank10_e_air",
    "diff_dayw"
]
DROPCOLS_ROUGHER = [
    "rougher.state.floatbank10_f_level"
]

COLS_TO_DIFF_TOP10 = set([
    # these matter for Final
    "rougher.input.feed_zn",
    "primary_cleaner.input.xanthate",
    "rougher.input.floatbank10_xanthate",
    "rougher.input.feed_pb",
    "primary_cleaner.input.depressant",
    "encod_val_rougher.input.feed_zn",
    "rougher.input.floatbank11_xanthate",
    "primary_cleaner.state.floatbank8_d_level"
    "rougher.input.floatbank10_copper_sulfate"
    "encod_val_rougher.input.feed_pb",
    "primary_cleaner.state.floatbank8_c_level"
    "rougher.input.feed_sol",
    "primary_cleaner.state.floatbank8_c_air"
    "primary_cleaner.input.copper_sulfate",
    # these come from Rougher (deleted the duplicates)
    "rougher.input.floatbank11_xanthate",
    "encod_val_rougher.input.feed_zn",
    "rougher.input.feed_fe",
    "rougher.state.floatbank10_d_air",
    "rougher.state.floatbank10_c_level",
    "encod_val_rougher.input.feed_pb",
    "encod_rel_rougher.input.feed_zn",
    "rougher.input.floatbank11_copper_sulfate",
    "rougher.input.feed_sol",
    "rougher.input.feed_size"
])
#
# COLS_TO_DIFF_TOP20 = [
#
# ]
level0_models_rougher = {}
obj = 'mae'
level0_models_rougher['LGBM_rougher_base_a'] = lgb.LGBMRegressor(objective=obj,
                                                                 learning_rate=0.05, n_estimators=500, random_state=91,
                                                                 **{'max_depth': 5, 'num_leaves': 100, 'feature_fraction': '0.363', 'bagging_fraction': '0.262'})
level0_models_rougher['LGBM_rougher_base_b'] =lgb.LGBMRegressor(objective=obj,
                                                                learning_rate=0.05, n_estimators=500, random_state=92,
                                                                **{'max_depth': 4, 'num_leaves': 110, 'feature_fraction': '0.448', 'bagging_fraction': '0.445'})
level0_models_rougher['LGBM_rougher_base_c'] =lgb.LGBMRegressor(objective=obj,
                                                                learning_rate=0.05, n_estimators=500, random_state=93,
                                                                **{'max_depth': 4, 'num_leaves': 155, 'feature_fraction': '0.449', 'bagging_fraction': '0.598'})
level0_models_rougher['LGBM_rougher_base_d'] =lgb.LGBMRegressor(objective=obj,
                                                                learning_rate=0.05, n_estimators=500, random_state=94,
                                                                **{'max_depth': 5, 'num_leaves': 210, 'feature_fraction': '0.472', 'bagging_fraction': '0.682'})
level0_models_rougher['LGBM_rougher_base_e']= lgb.LGBMRegressor(objective=obj,
                                                                learning_rate=0.05, n_estimators=500, random_state=7,
                                                                **{'max_depth': 5, 'num_leaves': 200, 'feature_fraction': '0.45', 'bagging_fraction': '0.72'})
level0_models_rougher['LGBM_rougher_base_f']= lgb.LGBMRegressor(objective=obj,
                                                                learning_rate=0.07, n_estimators=500, random_state=8,
                                                                **{'max_depth': 4, 'num_leaves': 63, 'feature_fraction': '0.879', 'bagging_fraction': '0.727'})
level0_models_rougher['LGBM_rougher_base_g']= lgb.LGBMRegressor(objective=obj,
                                                                learning_rate=0.07, n_estimators=500, random_state=9,
                                                                **{'max_depth': 5, 'num_leaves': 65, 'feature_fraction': '0.879', 'bagging_fraction': '0.727'})
level0_models_rougher['LGBM_rougher_base_h']= lgb.LGBMRegressor(objective=obj,
                                                                learning_rate=0.07, n_estimators=500, random_state=10,
                                                                **{'max_depth': 4, 'num_leaves': 60, 'feature_fraction': '0.797', 'bagging_fraction': '0.982'})
level0_models_rougher['LGBM_rougher_base_i'] = lgb.LGBMRegressor(objective=obj,
                                                                 learning_rate=0.07, n_estimators=500, random_state=12,
                                                                 **{'max_depth': 5, 'num_leaves': 60, 'feature_fraction': '0.8', 'bagging_fraction': '0.92'})
# level0_models['KNN_rougher_a'] = make_pipeline(scaler,KNeighborsRegressor(n_jobs = -1,**{'n_neighbors': 254, 'weights': 'distance', 'leaf_size': 16}))
scaler = make_pipeline(QuantileTransformer(output_distribution='normal'),PCA(whiten=True))
level0_models_rougher['KNN_rougher_b'] = make_pipeline(scaler, KNeighborsRegressor(n_jobs = -1, **{'n_neighbors': 50, 'weights': 'distance', 'leaf_size': 18}))
level0_models_rougher['KNN_rougher_c'] = make_pipeline(scaler, KNeighborsRegressor(n_jobs = -1, **{'n_neighbors': 15, 'weights': 'distance', 'leaf_size': 30.0}))
level0_models_rougher['KNN_rougher_d'] = make_pipeline(scaler, KNeighborsRegressor(n_jobs = -1, **{'n_neighbors': 5, 'weights': 'uniform', 'leaf_size': 24.0}))
level0_models_rougher['KNN_rougher_b_bray'] = make_pipeline(scaler, KNeighborsRegressor(n_jobs = -1, **{'n_neighbors': 50, 'weights': 'distance', 'metric': 'braycurtis', 'leaf_size': 18}))
level0_models_rougher['KNN_rougher_c_bray'] = make_pipeline(scaler, KNeighborsRegressor(n_jobs = -1, **{'n_neighbors': 15, 'weights': 'distance', 'leaf_size': 30.0, 'metric': 'braycurtis'}))
level0_models_rougher['KNN_rougher_d_bray'] = make_pipeline(scaler, KNeighborsRegressor(n_jobs = -1, **{'n_neighbors': 5, 'weights': 'uniform', 'leaf_size': 24.0, 'metric': 'braycurtis'}))



level0_models_final ={}
level0_models_final['LGBM_final_base_a'] = lgb.LGBMRegressor(objective='mae',
                              learning_rate=0.05, n_estimators=400,random_state=96,
                              **{'max_depth': 6, 'num_leaves': 10, 'feature_fraction': '0.411', 'bagging_fraction': '0.827'})

level0_models_final['LGBM_final_base_b'] =lgb.LGBMRegressor(objective='mae',
                              learning_rate=0.05, n_estimators=400,random_state=973,
                              **{'max_depth': 6, 'num_leaves': 15, 'feature_fraction': '0.515', 'bagging_fraction': '0.469'})

level0_models_final['LGBM_final_base_c'] =lgb.LGBMRegressor(objective='mae',
                              learning_rate=0.05, n_estimators=400,random_state=937,
                              **{'max_depth': 3, 'num_leaves': 195, 'feature_fraction': '0.635', 'bagging_fraction': '0.673'})
level0_models_final['LGBM_final_base_d'] =lgb.LGBMRegressor(objective='mae',
                              learning_rate=0.05, n_estimators=400,random_state=49,
                              **{'max_depth': 4, 'num_leaves': 70, 'feature_fraction': '0.795', 'bagging_fraction': '0.656'})

level0_models_final['KNN_final_b'] = make_pipeline(scaler,KNeighborsRegressor(n_jobs = -1,**{'n_neighbors': 50, 'weights': 'distance', 'leaf_size': 18}))
level0_models_final['KNN_final_c'] = make_pipeline(scaler,KNeighborsRegressor(n_jobs = -1,**{'n_neighbors': 15, 'weights': 'distance', 'leaf_size': 30.0}))
level0_models_final['KNN_final_d'] = make_pipeline(scaler,KNeighborsRegressor(n_jobs = -1,**{'n_neighbors': 5, 'weights': 'uniform', 'leaf_size': 24.0}))

level0_models_final['KNN_rougher_b_bray'] = make_pipeline(scaler,KNeighborsRegressor(n_jobs = -1,**{'n_neighbors': 50, 'weights': 'distance','metric':'braycurtis', 'leaf_size': 18}))
level0_models_final['KNN_rougher_c_bray'] = make_pipeline(scaler,KNeighborsRegressor(n_jobs = -1,**{'n_neighbors': 15, 'weights': 'distance', 'leaf_size': 30.0,'metric':'braycurtis'}))
level0_models_final['KNN_rougher_d_bray'] = make_pipeline(scaler,KNeighborsRegressor(n_jobs = -1,**{'n_neighbors': 5, 'weights': 'uniform', 'leaf_size': 24.0,'metric':'braycurtis'}))


