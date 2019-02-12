from model import XGBoost, QuantileGB, SVM, QuantileRF
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import pickle
from pathlib import Path


if __name__ == '__main__':
    import pickle
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    root = Path(__file__).resolve().parents[2]
    print(root)
    # Get raw features
    with open(f'{root}/data/processed/data_split_by_year_test_train.pkl', 'wb') as f:
        data_dict = pickle.load(f)
    # X_train is used for training and validation, X_test - final predictions (we have no labels for it)
    # Fix the year at 2016 for now
    X_train = data_dict[2016]['X_train']
    y_train = data_dict[2016]['y_train']
    print(f'X_train shape: {X_train.shape}, y_train: {y_train.shape}')
