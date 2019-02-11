import numpy as np

np.random.seed(123)
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn import neighbors
from sklearn.preprocessing import Normalizer


class Model(object):

    def evaluate(self, X_val, y_val):
        pred_target = self.guess(X_val)
        err = np.mean(np.abs((y_val - pred_target)))
        result = err
        return result


class LinearModel(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.clf = linear_model.LinearRegression()
        self.clf.fit(X_train, y_train)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        return self.clf.predict(feature)


class RF(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.clf = RandomForestRegressor(n_estimators=200, verbose=True, max_depth=35, min_samples_split=2,
                                         min_samples_leaf=2)
        self.clf.fit(X_train, y_train)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        return np.exp(self.clf.predict(feature))


class SVM(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.__normalize_data()
        self.clf = SVR(kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=0.001,
                       C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

        self.clf.fit(self.X_train, self.y_train)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def __normalize_data(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)

    def guess(self, feature):
        return self.clf.predict(feature)


class XGBoost(Model):

    def __init__(self, X_train, y_train, X_val, y_val,params = {'nthread': -1,
                 'max_depth': 10,
                 'eta': 0.2,
                 'objective': 'reg:linear',
                 'colsample_bytree': 0.7,
                 'subsample': 0.7},num_round = 300,esr = 30):
        super().__init__()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        evallist = [(dtrain, 'train'),(dvalid, 'valid')]




        self.bst = xgb.train(params, dtrain, num_round, evallist,early_stopping_rounds = esr)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        dtest = xgb.DMatrix(feature)
        return self.bst.predict(dtest)


class HistoricalMedian(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.history = {}
        self.feature_index = [1, 2, 3, 4]
        for x, y in zip(X_train, y_train):
            key = tuple(x[self.feature_index])
            self.history.setdefault(key, []).append(y)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = np.array(features)
        features = features[:, self.feature_index]
        hist_median = [np.median(self.history[tuple(feature)]) for feature in features]
        return np.array(hist_median)


class KNN(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.normalizer = Normalizer()
        self.normalizer.fit(X_train)
        self.clf = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance', p=1)
        self.clf.fit(self.normalizer.transform(X_train), y_train)
        print("Result on validation data: ", self.evaluate(self.normalizer.transform(X_val), y_val))

    def guess(self, feature):
        return self.clf.predict(self.normalizer.transform(feature))
#
# class NN(Model):
#
#     def __init__(self, X_train, y_train, X_val, y_val):
#         super().__init__()
#         self.epochs = 10
#         self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
#         self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
#         self.__build_keras_model()
#         self.fit(X_train, y_train, X_val, y_val)
#
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
#     def guess(self, features):
#         result = self.model.predict(features).flatten()
#         return self._val_for_pred(result)
