import sys

import pandas as pd

from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import random_uniform
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D,Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
import keras.backend as K
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from keras_tqdm import TQDMCallback
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split,cross_val_predict
from src.models.model import HistoricalMedian,XGBoost,LinearModel,RF,KNN,SVM,mase
from src.data.make_dataset import DROPCOLS
import numpy as np
import tensorflow as tf
sys.path.insert(0,'..')
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.pipeline import make_pipeline

sys.path.insert(0,'..')
from src.data.make_dataset import DROPCOLS
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


# window_step = 24 # in hours
# window_overlap = 12 # in hours
# Xm = X.values
# i=1
# st = window_step - window_overlap)
# window = Xm[window_step*(i):window_step*(i+1),].reshape(1,24,65)

def cut_cycle(data_cycle, overlap=0.8, base_length=150):
    step = int((1 - overlap) * base_length)
    n_windows = np.ceil((data_cycle.shape[0] - base_length) / step)
    indexer = np.arange(base_length)[None, :] + step * np.arange(n_windows)[:, None]
    return indexer.astype('int'), int(n_windows)


def produce_training_data(X, y, tgt='final.output.recovery', ov=0.9):
    a, b = cut_cycle(y, overlap=ov, base_length=24)
    Xwind = X[a, :]
    ywind = y.iloc[a[:, -1],]
    mask_weights = ((y[tgt] > 41) & (y[tgt] < 100))
    maskwind = mask_weights.iloc[a[:, -1],]
    ywind["mask"] = maskwind
    return Xwind, ywind

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)



root = Path(__file__).resolve().parents[2]

df_tsfresh = pd.read_pickle(f'{root}/data/processed/train_test_tsfresh.pkl').reset_index(level = 0)
data_dict = pd.read_pickle(f'{root}/data/processed/data_dict_all.pkl')

# For Final
tgt = 'final.output.recovery'

year = 2019
X = data_dict[year]['X_train'].astype(float).drop(DROPCOLS,axis = 1)
y = data_dict[year]['y_train'].astype(float)
# Add Lags:
mask_weights = ((y[tgt] > 41) & (y[tgt] < 100))
print(f'X = {X.shape} y = {y.shape}')
print(f'DROPNA X = {X.dropna().shape} DROPNA y = {y[tgt].dropna().shape}')



def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


def tilted_loss(y, f, q=0.55):
    e = (y - f)
    return K.mean(K.maximum(q * e, (q - 1) * e),
                  axis=-1)


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))


def create_model(optimizer=Adam(lr=0.01),
                 kernel_initializer=random_uniform(seed=12),
                 dropout=0.1, TIME_PERIODS=24, num_channels=65):
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=(TIME_PERIODS, num_channels), padding='same', strides=1))
    model.add(Conv1D(32, 3, activation='relu',padding = 'same',strides = 1))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu', kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=kernel_initializer))

    model.compile(loss=huber_loss, optimizer=optimizer, metrics=['mae'])

    return model


# wrap the model using the function you created
reg = KerasRegressor(build_fn=create_model, verbose=0)

scaler = make_pipeline(QuantileTransformer(output_distribution='normal'), StandardScaler(), PCA(whiten=True))
target_scaler = make_pipeline(QuantileTransformer(output_distribution='normal'), StandardScaler())

callbacks = [EarlyStopping(monitor='val_loss', patience=6),
             ModelCheckpoint(filepath=f'{root}/notebooks/keras-ch/best_model-window.h5', monitor='val_loss', save_best_only=True)]
n_folds = 6
cv = KFold(n_folds, shuffle=False, random_state=42)
model = reg
scores = []





fig,ax = plt.subplots(figsize = (20,16),nrows = n_folds)
X_new = scaler.fit_transform(X)
Xw,yw = produce_training_data(X_new,y,ov=0.95)
mask = yw['mask'].values
yw[tgt] = yw[tgt].fillna(method='bfill')
yf = yw[tgt].values
preds_all_alt = np.empty_like(yf)
preds_all_base = np.empty_like(yf)
true_all =np.empty_like(yf)
history = [None for i in range(n_folds)]

for fold_n, (train_index, valid_index) in enumerate(cv.split(yf)):
    # print('Fold', fold_n, 'started at', time.ctime())
    print(train_index)
    X_train, X_valid = Xw[train_index, :, :], Xw[valid_index, :, :]
    mask_train, mask_valid = mask[train_index],mask[valid_index]
    y_train, y_valid = yf[train_index], yf[valid_index]
    # Do the base
    params = {"validation_data": (X_valid, y_valid),
              "epochs": 30,
              "verbose": 1,
              "batch_size": 8,
              "callbacks": callbacks,
              'sample_weight':mask_train}
    model = KerasRegressor(build_fn=create_model, verbose=0)
    history[fold_n] = model.fit(X_train, y_train, **params)
    preds = model.predict(X_valid)
    preds_all_base[valid_index] = preds
    true_all[valid_index] = y_valid
    score_val = mase(preds, y_valid)
    df = pd.DataFrame({"preds": preds, "true": y_valid})
    df.plot(ax=ax[fold_n], style=['-o', '-o', '-o'], title=f'CV score base: {score_val:.4f}', markersize=1.5)
fig.savefig('window-keras.png')
oof_scores = mase(preds_all_base, true_all)
print(oof_scores)