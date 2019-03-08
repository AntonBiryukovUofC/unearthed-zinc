# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import zipfile

DROPCOLS = ['rougher.input.floatbank11_copper_sulfate',
            'rougher.input.floatbank10_xanthate',
            'rougher.state.floatbank10_c_air',
            'rougher.state.floatbank10_e_air',
            'rougher.state.floatbank10_f_air',
            'rougher.state.floatbank10_d_level',
            'rougher.state.floatbank10_f_level',
            'primary_cleaner.state.floatbank8_b_air',
            'primary_cleaner.state.floatbank8_c_air',
            "secondary_cleaner.state.floatbank4_b_air",
            'secondary_cleaner.state.floatbank2_b_air',
            "secondary_cleaner.state.floatbank5_b_air",
            "secondary_cleaner.state.floatbank3_a_air"
            ]


def to_integer(dt_time):
    y = 365 * (dt_time.year - 2016) + 30.25 * dt_time.month + dt_time.day
    return y.to_numpy().reshape(-1, 1)


def get_ransac(y=None, threshold=30):
    from sklearn import linear_model
    rs = linear_model.RANSACRegressor()
    y = y[y > threshold]
    y_rs = y.rolling(window=24 * 7).min().dropna()
    t = to_integer(y_rs.index)
    rs.fit(t, y_rs)
    return rs


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
    # df_train['date'] =df_train['date'].dt.round('H')
    data_dict = pd.read_json(f'{root}/data/interim/data_dictionary_v1.json')
    df_test = pd.read_csv(f'{root}/data/interim/test_data/all_test.csv', parse_dates=['date'])
    # df_test['date'] = df_test['date'].dt.round('H')

    # Concatenate , index by test-train
    cols = df_test.columns
    tgts = ['rougher.output.recovery', 'final.output.recovery', 'secondary_cleaner.output.tail_zn',
            'final.output.tail_zn',
            'final.output.concentrate_zn', 'primary_cleaner.output.concentrate_zn', 'rougher.output.concentrate_zn']
    target_df = df_train[tgts + ['date']].set_index('date')

    # Fill NAs by propagating last available value
    df_all = pd.concat([df_train[cols], df_test], keys=['train', 'test']).fillna(method='ffill')
    print(f'Train shape = {df_train.shape} , test shape = {df_test.shape}')
    # Build ransac for trends:
    # rs_rough = get_ransac(target_df[tgts[0]], threshold=45)
    # rs_final = get_ransac(target_df[tgts[1]], threshold=45)
    # Calculate features from ransac:
    # ransac_rough = pd.DataFrame({'rs_rough':rs_rough.predict(to_integer(df_all['date'].dt))},index = df_all['date'])
    # ransac_final= pd.DataFrame({'rs_final':rs_final.predict(to_integer(df_all['date'].dt))},index = df_all['date'])

    # cols_10_floatbank_air =[f'rougher.state.floatbank10_{i}_air' for i in ['bcdef']]
    # cols_10_floatbank_lvl = [f'rougher.state.floatbank10_{i}_level' for i in ['abcdef']]

    # cols_8_floatbank_air = [f'primary_cleaner.state.floatbank8_{i}_air' for i in ['abcd']]
    # cols_8_floatbank_lvl = [f'primary_cleaner.state.floatbank8_{i}_level' for i in ['abcd']]

    # df_culled['rougher.state.floatbank10_mean_air'] = df_culled[cols_10_floatbank_air].mean(axis=1)
    # df_culled['rougher.state.floatbank10_mean_level'] = df_culled[cols_10_floatbank_lvl].mean(axis=1)

    # df_culled['rougher.state.floatbank8_mean_air'] = df_culled[cols_8_floatbank_air].mean(axis=1)
    # df_culled['rougher.state.floatbank8_mean_level'] = df_culled[cols_8_floatbank_lvl].mean(axis=1)

    # df_all['daym'] = df_all['date'].dt.day / df_all['date'].dt.daysinmonth
    day_r = target_df.groupby(target_df.index.day)['rougher.output.recovery'].mean().to_dict()
    day_f = target_df.groupby(target_df.index.day)['final.output.recovery'].mean().to_dict()

    df_all['daily_avg_rougher'] = df_all['date'].dt.day.apply(lambda x: day_r[x])
    df_all['daily_avg_final'] = df_all['date'].dt.day.apply(lambda x: day_f[x])

    df_all['hour'] = df_all['date'].dt.hour
    df_all['dayw'] = df_all['date'].dt.day

    df_all['week'] = df_all['date'].dt.week % 4
    df_all = df_all.set_index('date', append=True)
    # Merge with temperatures:
    temps = pd.read_csv(f'{root}/data/raw/min_temp.csv')
    temps = temps[temps.Year >= 2016]
    temps['date'] = pd.to_datetime(temps[['Year', 'Month', 'Day']])
    temps = temps.set_index('date').resample('1H').pad()
    temps = temps.tz_localize('UTC')
    temps['temp'] = temps['Minimum temperature (Degree C)']
    # df_all = df_all.join(temps['temp'])
    # df_all['temp'] = df_all['temp'].fillna(method='ffill')

    # merge with ransac:

    # df_all = df_all.join(ransac_rough)
    # df_all = df_all.join(ransac_final)
    df_all_with_tgt = df_all.copy().join(target_df)
    df_all_with_tgt.to_pickle(f'{root}/data/processed/train_test_raw.pkl')
    # Prepare PCA here:
    pc = PCA().fit(df_all)
    nind = (np.cumsum(pc.explained_variance_ratio_) <= 0.9999).sum()
    pc_cols = [f'PC{i}' for i in range(nind)]
    pc_df = pd.DataFrame(data=pc.transform(df_all)[:, :nind], columns=pc_cols, index=df_all.index).reset_index(level=1,
                                                                                                               drop=True)
    pc_df.to_pickle(f'{root}/data/processed/train_test_PCA.pkl')

    # Get tsfresh:
    p_l = 12
    p_s = 6
    p_m = 24
    p_sl = 24 * 5
    df_culled = df_all.copy()
    df_culled = df_culled.drop(DROPCOLS, axis=1)

    df_tsfresh_feats_med = get_tsfresh_features(df_culled, max_timeshift=p_m)
    df_tsfresh_feats_med.to_pickle(f'{root}/data/processed/train_test_tsfresh_{p_m}.pkl')

    df_tsfresh_feats_long = get_tsfresh_features(df_culled, max_timeshift=p_l)
    df_tsfresh_feats_long.to_pickle(f'{root}/data/processed/train_test_tsfresh_{p_l}.pkl')

    df_tsfresh_feats_short = get_tsfresh_features(df_culled, max_timeshift=p_s)
    df_tsfresh_feats_short.to_pickle(f'{root}/data/processed/train_test_tsfresh_{p_s}.pkl')

    df_tsfresh_feats_sl = get_tsfresh_features(df_culled, max_timeshift=p_sl)
    df_tsfresh_feats_sl.to_pickle(f'{root}/data/processed/train_test_tsfresh_{p_sl}.pkl')


def get_tsfresh_features(df=None, max_timeshift=10, n_jobs=10):
    from tsfresh.utilities.dataframe_functions import make_forecasting_frame
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import extract_features
    import pandas as pd
    if max_timeshift > 10:
        d = {'skewness': None,
             'kurtosis': None,
             'quantile': [{'q': 0.05},
                          {'q': 0.95}],
             'linear_trend': [
                 {'attr': 'slope'}
             ],
             'mean_abs_change': None,
             'mean_second_derivative_central': None,
             'fft_aggregated': [
                 {'aggtype': "centroid"},
                 {'aggtype': "variance"},
                 {'aggtype': "skew"},
                 {'aggtype': "kurtosis"}
             ],
             'max_min_diff': None,
             'max_slope': None,
             'min_slope': None
             }
    else:
        d = {'mean': None,
             'maximum': None,
             'minimum': None,
             'mean_abs_change': None,
             'mean_second_derivative_central': None,
             'max_min_diff': None,
             'max_slope': None,
             'min_slope': None
             }

    df = df.fillna(method='ffill')
    df_tsfresh = df.reset_index(level=[0, 1], drop=True)
    dfs = {}
    cols_to_calc = ["rougher.input.feed_fe",
                    "rougher.input.feed_zn",
                    "rougher.input.feed_sol",
                    "rougher.input.feed_pb",
                    "rougher.input.feed_rate",
                    'rougher.input.floatbank11_xanthate',
                    'rougher.input.floatbank10_copper_sulfate',
                    'rougher.state.floatbank10_b_air',
                    "secondary_cleaner.state.floatbank5_a_air",
                    "primary_cleaner.input.copper_sulfate",
                    "primary_cleaner.state.floatbank8_a_air",
                    "primary_cleaner.input.depressant",
                    "primary_cleaner.input.feed_size",
                    "primary_cleaner.input.xanthate"
                    ]

    for c in cols_to_calc:
        print(f'Working on {c}...')
        df_shift, y = make_forecasting_frame(df_tsfresh[c], kind="price", max_timeshift=max_timeshift,
                                             rolling_direction=1)
        X = extract_features(df_shift, column_id="id", column_sort="time", column_value="value",
                             impute_function=impute, show_warnings=False, default_fc_parameters=d, n_jobs=n_jobs)
        dfs[c] = X
    df_tsfresh_feats = pd.concat(dfs, keys=list(dfs.keys()))
    df_tsfresh_feats.columns = [f'{i}_p{max_timeshift}' for i in df_tsfresh_feats.columns]
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
