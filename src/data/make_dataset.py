# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import zipfile

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


def encode(data, col, max_val):
    import numpy as np
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    return data


def main(root=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    import pandas as pd
    from sklearn.decomposition import PCA
    import numpy as np
    pca_threshold = 0.996

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    zip_ref = zipfile.ZipFile(f'{root}/data/raw/for_competition_release.zip', 'r')
    zip_ref.extractall(f'{root}/data/interim/')
    df_train = pd.read_csv(f'{root}/data/interim/train_data/all_train.csv', parse_dates=['date'])
    data_dict = pd.read_json(f'{root}/data/interim/data_dictionary_v1.json')
    df_test = pd.read_csv(f'{root}/data/interim/test_data/all_test.csv', parse_dates=['date'])
    # Concatenate , index by test-train
    cols = df_test.columns
    tgts = ['final.output.recovery','rougher.output.recovery']
    target_df = df_train[tgts + ['date']].set_index('date')
    # Fill NAs by propagating last available value
    df_all = pd.concat([df_train[cols], df_test], keys=['train', 'test']).fillna(method='ffill')
    print(f'Train shape = {df_train.shape} , test shape = {df_test.shape}')
    # We need to time-dependent features into cyclical:














    df_all['day'] = df_all['date'].dt.day / df_all['date'].dt.daysinmonth

    df_all = df_all.set_index('date',append = True)

    df_all_with_tgt = df_all.copy().join(target_df)
    df_all_with_tgt.to_pickle(f'{root}/data/processed/train_test_raw.pkl')
    # Prepare PCA here:
    pc = PCA().fit(df_all)
    nind = (np.cumsum(pc.explained_variance_ratio_) <=0.997).sum()
    pc_cols = [f'PC{i}' for i in range(nind)]
    pc_df = pd.DataFrame(data = pc.transform(df_all)[:,:nind],columns=pc_cols,index=df_all.index).reset_index(level=1,drop=True)
    pc_df.to_pickle(f'{root}/data/processed/train_test_PCA.pkl')

    # Get tsfresh:

    df_tsfresh_feats = get_tsfresh_features(df_all,max_timeshift=24)
    df_tsfresh_feats.to_pickle(f'{root}/data/processed/train_test_tsfresh.pkl')


def get_tsfresh_features(df=None, max_timeshift=10, n_jobs=10):
    from tsfresh.utilities.dataframe_functions import make_forecasting_frame
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import extract_features
    import pandas as pd
    d = {'median': None,
         'mean': None,
         'standard_deviation': None,
         'skewness': None,
         'kurtosis': None,
         'quantile': [{'q': 0.15},
                      {'q': 0.9}],
         'linear_trend': [
             {'attr': 'rvalue'},
             {'attr': 'intercept'},
             {'attr': 'slope'}
         ]}

    df = df.fillna(method='ffill')
    df_tsfresh = df.reset_index(level=[0,1],drop = True)
    dfs = {}
    cols_to_calc = ["rougher.input.feed_fe",
    "rougher.input.feed_zn",
    "secondary_cleaner.state.floatbank5_a_air",
    "primary_cleaner.input.copper_sulfate",
    "rougher.input.feed_sol",
    "primary_cleaner.state.floatbank8_a_air",
    "rougher.input.floatbank11_xanthate",
    "rougher.input.feed_pb",
    "primary_cleaner.state.floatbank8_b_air",
    "primary_cleaner.input.depressant"]


    for c in cols_to_calc:
        print(f'Working on {c}...')
        df_shift, y = make_forecasting_frame(df_tsfresh[c], kind="price", max_timeshift=max_timeshift,
                                             rolling_direction=1)
        X = extract_features(df_shift, column_id="id", column_sort="time", column_value="value",
                             impute_function=impute, show_warnings=False, default_fc_parameters=d, n_jobs=n_jobs)
        dfs[c] = X
    df_tsfresh_feats = pd.concat(dfs, keys=list(dfs.keys()))
    return df_tsfresh_feats


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main(root=project_dir)
