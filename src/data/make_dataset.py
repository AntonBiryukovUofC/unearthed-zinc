# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import zipfile

def main(root = None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    import pandas as pd
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    zip_ref = zipfile.ZipFile(f'{root}/data/raw/for_competition_release.zip','r')
    zip_ref.extractall(f'{root}/data/interim/')
    df_train = pd.read_csv(f'{root}/data/interim/train_data/all_train.csv',parse_dates=['date'])
    data_dict = pd.read_json(f'{root}/data/interim/data_dictionary_v1.json')
    df_test = pd.read_csv(f'{root}/data/interim/test_data/all_test.csv',parse_dates=['date'])
    # Concatenate , index by test-train
    cols = df_test.columns
    # Fill NAs by propagating last available value
    df_all = pd.concat([df_train[cols],df_test],keys = ['train','test']).fillna(method = 'ffill')
    print(f'Train shape = {df_train.shape} , test shape = {df_test.shape}')
    
    
    
    df_all['hour'] = df_all['date'].dt.hour
    df_all['dow'] = df_all['date'].dt.dayofweek
    df_all['day'] = df_all['date'].dt.day / df_all['date'].dt.daysinmonth
    df_all['month'] = df_all['date'].dt.month
    df_all['weekofyear'] = df_all['date'].dt.weekofyear
    df_all.to_csv(f'{root}/data/processed/train_test_raw.csv',index=False)
    # Get tsfresh:
    
    df_tsfresh_feats = get_tsfresh_features(df_all.drop(['hour','dow','day','month','weekofyear'],axis = 1))
    df_tsfresh_feats.to_csv(f'{root}/data/processed/train_test_tsfresh.csv')
    
    
def get_tsfresh_features(df = None,max_timeshift = 3*10,n_jobs = 4,every_n_hour = 4):
    from tsfresh.utilities.dataframe_functions import make_forecasting_frame
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import extract_features, MinimalFCParameters,EfficientFCParameters
    d = EfficientFCParameters()
    
    del d['friedrich_coefficients']
    del d['max_langevin_fixed_point']
    del d['autocorrelation']
    del d['agg_autocorrelation']
    del d['cwt_coefficients']
    del d['large_standard_deviation']
    del d['symmetry_looking']
    del d['partial_autocorrelation']
    del d['index_mass_quantile']
    del d['spkt_welch_density']
    del d['change_quantiles']
    del d['fft_aggregated']
    del d['fft_coefficient']
    del d['agg_linear_trend']

    
    df = df.fillna(method = 'ffill')
    df['hour'] = df['date'].dt.hour

    df_tsfresh = df.reset_index(drop=True).set_index('date')
    df_tsfresh = df_tsfresh[(df_tsfresh.hour % every_n_hour) == 0].drop('hour',axis = 1)
    dfs = {}
    for c in df_tsfresh.columns:
        print(f'Working on {c}...')
        df_shift, y = make_forecasting_frame(df_tsfresh[c], kind="price", max_timeshift=max_timeshift, rolling_direction=1)
        X = extract_features(df_shift, column_id="id", column_sort="time", column_value="value",
                             impute_function=impute,show_warnings=False, default_fc_parameters=d,n_jobs=n_jobs)
        dfs[c] = X
    df_tsfresh_feats = pd.concat(dfs,keys = list(dfs.keys()))
    return df_tsfresh_feats

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main(root = project_dir)
