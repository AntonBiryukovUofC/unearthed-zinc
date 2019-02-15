import numpy as np

np.random.seed(123)
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from skgarden import RandomForestQuantileRegressor,MondrianForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn import neighbors
from sklearn.preprocessing import Normalizer
import keras
from sklearn.metrics import make_scorer

from xgboost.sklearn import XGBRegressor

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

class Model(object):

    def evaluate(self, X_val, y_val):
        pred_target = self.predict(X_val)
        err = mase(y_val, pred_target)
        result = err
        return result


class LinearModel(Model):

    def __init__(self, X_train, y_train, X_val, y_val, params={"l1_ratio": 0.5, "alpha": 1e0}):
        super().__init__()
        self.model = linear_model.ElasticNet(**params)
        self.model.fit(X_train, y_train)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def predict(self, feature):
        return self.model.predict(feature)

    def fit_final(self, X_train, y_train, params):
        self.model.fit(X_train, y_train)

    def score(self, X_val, y_val):
        return self.evaluate(X_val, y_val)


class RF(Model):

    def __init__(self, X_train, y_train, X_val, y_val, kwargs={}):
        super().__init__()
        self.clf = RandomForestRegressor(**kwargs)
        self.clf.fit(X_train, y_train)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def predict(self, feature):
        return self.clf.predict(feature)


class QuantileRF(Model):

    def __init__(self, X_train, y_train, X_val, y_val, params={}, quantile=50):
        super().__init__()
        self.quantile = quantile
        self.model = RandomForestQuantileRegressor(**params)
        self.model.fit(X_train, y_train)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def fit_final(self, X_train, y_train, params):
        self.model.fit(X_train, y_train)

    def predict(self, feature):
        return self.model.predict(feature, self.quantile)

    def score(self, X_val, y_val):
        return self.evaluate(X_val, y_val)



class MondrianRF(Model):

    def __init__(self, X_train, y_train, X_val, y_val, params={}):
        super().__init__()
        self.model = MondrianForestRegressor(**params)
        self.model.fit(X_train, y_train)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def fit_final(self, X_train, y_train, params):
        self.model.fit(X_train, y_train)

    def predict(self, feature):
        return self.model.predict(feature)

    def score(self, X_val, y_val):
        return self.evaluate(X_val, y_val)

class QuantileGB(Model):
    def __init__(self, X_train, y_train, X_val, y_val, params={'n_estimators = 100'}, quantile=50):
        super().__init__()
        from sklearn.ensemble import GradientBoostingRegressor
        self.quantile = quantile
        self.model = GradientBoostingRegressor(loss='quantile', **params)
        self.model.fit(X_train, y_train)

        # print("Result on validation data: ", self.evaluate(X_val, y_val))

    def fit_final(self, X_train, y_train, params):
        self.model.fit(X_train, y_train)

    def predict(self, feature):
        return self.model.predict(feature)

    def score(self, X_val, y_val):
        return self.evaluate(X_val, y_val)


class SVM(Model):

    def __init__(self, X_train, y_train, X_val, y_val, kwargs={}):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.__normalize_data()
        self.clf = SVR(**kwargs)

        self.clf.fit(self.X_train, self.y_train)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def __normalize_data(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)

    def predict(self, feature):
        return self.clf.predict(feature)


class XGBoost(Model):

    def __init__(self, X_train, y_train, X_val, y_val, params={'nthread': -1,
                                                               'max_depth': 10,
                                                               'eta': 0.2,
                                                               'objective': 'reg:linear',
                                                               'colsample_bytree': 0.7,
                                                               'subsample': 0.7,
                                                               'num_round': 300,
                                                               'early_stopping_rounds': 100,
                                                               'verbose_eval': 50
                                                               }):
        super().__init__()

        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dvalid = xgb.DMatrix(X_val, label=y_val)
        self.evallist = [(self.dtrain, 'train'), (self.dvalid, 'valid')]

        self.bst = xgb.train(params, self.dtrain, params['num_round'], self.evallist,
                             early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=params['verbose_eval'])
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def fit_final(self, X_train, y_train, params):
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        evallist = [(self.dtrain, 'train')]
        self.bst = xgb.train(params, self.dtrain, params['num_round'], evallist,
                             early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=params['verbose_eval'])

    def predict(self, feature, params={}):
        dtest = xgb.DMatrix(feature)
        return self.bst.predict(dtest, **params)

    def score(self, X_val, y_val):
        return self.evaluate(X_val, y_val)


class HistoricalMedian(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.history = {}
        self.feature_index = [1, 2, 3, 4]
        for x, y in zip(X_train, y_train):
            key = tuple(x[self.feature_index])
            self.history.setdefault(key, []).append(y)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def predict(self, features):
        features = np.array(features)
        features = features[:, self.feature_index]
        hist_median = [np.median(self.history[tuple(feature)]) for feature in features]
        return np.array(hist_median)


class KNN(Model):

    def __init__(self, X_train, y_train, X_val, y_val, params={"n_neighbors": 10, "weights": 'distance', "p": 1}):
        super().__init__()
        self.normalizer = Normalizer()
        self.normalizer.fit(X_train)
        self.model = neighbors.KNeighborsRegressor(**params)
        self.model.fit(self.normalizer.transform(X_train), y_train)
        print("Result on validation data: ", self.evaluate(self.normalizer.transform(X_val), y_val))

    def predict(self, feature):
        return self.model.predict(self.normalizer.transform(feature))

    def fit_final(self, X_train, y_train, params):
        self.model.fit(self.normalizer.transform(X_train), y_train)

    def score(self, X_val, y_val):
        return self.evaluate(X_val, y_val)


def tilted_loss(q, y, f):
    e = (y - f)
    return keras.backend.mean(keras.backend.maximum(q * e, (q - 1) * e),
                              axis=-1)


class QuantileKeras(Model):

    def __init__(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, quantile=0.5, input_dim=400,
                 batch_size=128):
        super().__init__()
        self.epochs = epochs
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.checkpointer = keras.callbacks.ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1,
                                                            save_best_only=True)
        self.early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        self.__build_keras_model(lr)
        X_train = np.expand_dims(X_train, 1)
        X_val = np.expand_dims(X_val, 1)
        self.quantile = quantile
        self.max_y = max(np.max(y_train), np.max(y_val))
        self.fit(X_train, y_train, X_val, y_val)

    def __build_keras_model(self, lr):
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        from keras.optimizers import Adam
        self.model = Sequential()
        self.model.add(Dense(1000, kernel_initializer="uniform", input_dim=self.input_dim))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, kernel_initializer="uniform"))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss=lambda y, f: tilted_loss(self.quantile, y, f), optimizer=Adam(lr=lr))

    def _val_for_fit(self, val):
        val = np.log(val) / np.log(self.max_y)
        return val

    def _val_for_pred(self, val):
        return np.exp(val) * self.max_y

    #
    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, self._val_for_fit(y_train),
                       validation_data=(X_val, self._val_for_fit(y_val)),
                       epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=[self.checkpointer, self.early_stop], verbose=1)
        # self.model.load_weights('best_model_weights.hdf5')
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def predict(self, features):
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)


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

def mase_error(y_true,y_pred):
    return mase(y_pred,y_true)

def my_custom_scorer():
    return make_scorer(mase_error, greater_is_better=False)

#     def __build_keras_model(self):
#         self.model = Sequential()
#         self.model.add(Dense(1000, kernel_initializer="uniform", input_dim=1183))
#         self.model.add(Activation('relu'))
#         self.model.add(Dense(500, kernel_initializer="uniform"))
#         self.model.add(Activation('relu'))
#         self.model.add(Dense(1))
#         self.model.add(Activation('sigmoid'))
#
#         self.model.compile(loss='mean_absolute_error', optimizer='adam')
#
#     def _val_for_fit(self, val):
#         val = numpy.log(val) / self.max_log_y
#         return val
#
#     def _val_for_pred(self, val):
#         return numpy.exp(val * self.max_log_y)
#
#     def fit(self, X_train, y_train, X_val, y_val):
#         self.model.fit(X_train, self._val_for_fit(y_train),
#                        validation_data=(X_val, self._val_for_fit(y_val)),
#                        epochs=self.epochs, batch_size=128,
#                        # callbacks=[self.checkpointer],
#                        )
#         # self.model.load_weights('best_model_weights.hdf5')
#         print("Result on validation data: ", self.evaluate(X_val, y_val))
#
#     def predict(self, features):
#         result = self.model.predict(features).flatten()
#         return self._val_for_pred(result)


def train_model(X, y, folds, model=None, score=mase, params=None, fixed_length=False, train_splits=None,
                test_splits=None):
    import time
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numpy as np


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
