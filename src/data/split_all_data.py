import logging
from pathlib import Path
import pandas as pd
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline


def clean_output_outliers(df):
    # 'secondary_cleaner.output.tail_zn', 'final.output.tail_zn',
    # 'final.output.concentrate_zn', 'primary_cleaner.output.concentrate_zn', 'rougher.output.concentrate_zn']
    # pass
    print(f'Shape Y before : {df.shape}')
    mask = (df['rougher.output.recovery'] > 45) & (df['rougher.output.recovery'] < 100) & \
           (df['final.output.recovery'] > 35) & (df['final.output.recovery'] < 100)
    # (df['final.output.tail_zn'] > 0.8) & \
    # (df['final.output.concentrate_zn'] > 30) & \
    # (df['primary_cleaner.output.concentrate_zn'] > 10) & \
    # (df['rougher.output.concentrate_zn'] > 8) & \
    # (df['secondary_cleaner.output.tail_zn'] > 1)
    # (df['final.output.recovery'] < df['rougher.output.recovery']) & \

    return mask


def clean_input_outliers(df):
    # 'secondary_cleaner.output.tail_zn', 'final.output.tail_zn',
    # 'final.output.concentrate_zn', 'primary_cleaner.output.concentrate_zn', 'rougher.output.concentrate_zn']
    # pass
    print(f'Shape Y before : {df.shape}')
    df = df[
        (df['primary_cleaner.input.feed_size'] > 5.50) &
        (df['primary_cleaner.input.xanthate'] > 0.05) & (df['primary_cleaner.input.xanthate'] < 2.0) &
        (df['primary_cleaner.state.floatbank8_a_level'] > -520) & (
                df['primary_cleaner.state.floatbank8_a_level'] < -360) &
        (df['primary_cleaner.state.floatbank8_c_air'] > 1000) &
        (df['primary_cleaner.state.floatbank8_c_level'] > -530) &
        (df['rougher.input.feed_rate'] > 250) &
        (df['rougher.input.feed_size'] < 115) &
        (df['rougher.input.floatbank11_copper_sulfate'] < 20.50) & (
                df['rougher.input.floatbank11_copper_sulfate'] > 0.50) &
        (df['rougher.input.floatbank11_xanthate'] > 2.0) &
        (df['rougher.state.floatbank10_d_air'] > 780) &
        (df['secondary_cleaner.state.floatbank2_a_air'] > 22)
        ]
    return df


def return_umap(X=None, y=None, X_test=None, n_comp=3, n_neighbors=15, metric='euclidean'):
    from sklearn.decomposition import PCA
    import umap
    reducer = make_pipeline(PCA(whiten=True),
                            umap.UMAP(n_neighbors=n_neighbors, metric=metric, random_state=123, n_components=n_comp))
    umap_obj = reducer.fit(X, y=y)
    uX_train = umap_obj.transform(X)
    uX_test = umap_obj.transform(X_test)

    cols = [f'PC{i}' for i in range(n_comp)]
    u_df_train = pd.DataFrame(data=uX_train, columns=cols, index=X.index)
    u_df_test = pd.DataFrame(data=uX_test, columns=cols, index=X_test.index)

    return u_df_train, u_df_test


def split_data(root):
    # TSFRESH
    df_flats = []
    for p in [12, 6, 24, 24 * 5]:
        df_tsfresh = pd.read_pickle(f'{root}/data/processed/train_test_tsfresh_{p}.pkl').reset_index(level=0)
        tmp = df_tsfresh.pivot_table(index='id', columns=['level_0'])

        # tmp.columns = ['_'.join(col).strip() for col in tmp.columns.values]
        df_flats.append(tmp)

    # Create metafeatures using rolling min-max:

    df_flat = pd.concat(df_flats, axis=1)

    df_flat.index.names = ['date']
    df_flat.columns = ['_'.join(col).strip() for col in df_flat.columns.values]
    df_flat.columns = [x.replace("\"", "") for x in df_flat.columns]

    # df_flat = create_meta_features(df_flat)

    df_excl = pd.read_csv(f'{root}/data/external/distr-shifted.csv', parse_dates=['date'])

    df_all = pd.read_pickle(f'{root}/data/processed/train_test_raw.pkl')
    df_pca = pd.read_pickle(f'{root}/data/processed/train_test_PCA.pkl')

    data_dict = {}
    years = [2016, 2017, 2018]
    tgts = ['rougher.output.recovery', 'final.output.recovery', 'secondary_cleaner.output.tail_zn',
            'final.output.tail_zn',
            'final.output.concentrate_zn', 'primary_cleaner.output.concentrate_zn', 'rougher.output.concentrate_zn']

    print('Encoding features')
    df_encoded = encode_imp_features(df=df_all)
    df_all = df_all.join(df_encoded)

    #print('Calculating the lag differences')
    #df_lag_2 = calculate_lagdiffs_features(df=df_all.copy(), base_lag=0, step_lag=2, cols_to_diff=COLS_TO_DIFF_TOP10)
    #df_lag_3 = calculate_lagdiffs_features(df=df_all.copy(), base_lag=1, step_lag=3, cols_to_diff=COLS_TO_DIFF_TOP10)
    #df_lag_rel = calculate_lagdiffs_features(df=df_all.copy(), base_lag=0, step_lag=3, cols_to_diff=COLS_TO_DIFF_TOP10,
    #                                        relative=True)

    #print("Calculate the derivatives of N Order")
    #df_deriv_1 = calculate_derivative_features(df=df_all.copy(), order=1, cols_to_diff=COLS_TO_DIFF_TOP10)
    #df_deriv_2 = calculate_derivative_features(df=df_all.copy(), order=2, cols_to_diff=COLS_TO_DIFF_TOP10)
    #df_deriv_3 = calculate_derivative_features(df=df_all.copy(), order=3, cols_to_diff=COLS_TO_DIFF_TOP10)
    # df_deriv_4 = calculate_derivative_features(df=df_all.copy(), order=4, cols_to_diff=COLS_TO_DIFF_TOP10)

    #print('Concatenating lags')
    #df_all = pd.concat([df_all, df_lag_2, df_lag_3, df_lag_rel, df_deriv_1, df_deriv_2, df_deriv_3], axis=1)
    df_all = df_all.reset_index('date')
    print('Splitting by year')

    for y in years:
        inds = df_all['date'].dt.year == y

        X_train = df_all[inds].loc['train',].drop(tgts, axis=1).set_index('date', append=True).reset_index(level=0,
                                                                                                           drop=True)
        X_train_ts = X_train.join(df_flat, how='inner')
        X_train_ts = create_meta_features(X_train_ts)

        #interaction_train = calculate_interactions_poly(X_train_ts)
        #interaction_diffrat_train =calculate_interactions_diff_ratio(X_train_ts)

        #X_train_ts = pd.concat([X_train_ts, interaction_train,interaction_diffrat_train], axis=1)

        X_train_pca = df_pca.loc[('train',), :].loc[X_train.index]

        y_train = df_all[inds].loc['train',][tgts + ['date']].set_index('date', append=True).reset_index(level=0,
                                                                                                         drop=True)
        if y != 2018:
            X_test = df_all[inds].loc['test',].drop(tgts, axis=1).set_index('date', append=True).reset_index(level=0,
                                                                                                             drop=True)
            X_test_ts = X_test.join(df_flat, how='inner')
            #X_test_ts = create_meta_features(X_test_ts)

            #interaction_test = calculate_interactions_poly(X_test_ts)
            #interaction_diffrat_test = calculate_interactions_diff_ratio(X_test_ts)

            #X_test_ts = pd.concat([X_test_ts, interaction_test,interaction_diffrat_test], axis=1)

            X_test_pca = df_pca.loc[('test',),].loc[X_test.index]
            # encoded_df_test = encode_imp_features(df=X_test)
            # X_test = X_test.join(encoded_df_test)
            # X_test_ts = X_test_ts.join(encoded_df_test)
        else:
            X_test = None
            X_test_pca = None
        # Plug- in the encodings:

        data_dict[y] = {'X_train': X_train, 'X_test': X_test, "y_train": y_train, 'X_train_ts': X_train_ts,
                        'X_test_ts': X_test_ts, 'X_train_pca': X_train_pca, 'X_test_pca': X_test_pca}

    X_to_clean = pd.concat(
        [data_dict[2016]['X_train_ts'], data_dict[2017]['X_train_ts'], data_dict[2018]['X_train_ts']])
    y_to_clean = pd.concat([data_dict[2016]['y_train'], data_dict[2017]['y_train'], data_dict[2018]['y_train']])

    mask = clean_output_outliers(y_to_clean)
    # X_to_clean = clean_input_outliers(X_to_clean)

    print(f'Shape Y after : {y_to_clean.shape}')
    print(f'Shape X after : {X_to_clean.shape}')

    inds = X_to_clean.index.intersection(y_to_clean.index)
    X_to_clean = X_to_clean.loc[inds, :]
    y_to_clean = y_to_clean.loc[inds, :]
    mask = mask.loc[inds]

    print(f'Shape X after intersect: {X_to_clean.shape}')

    # Calculate the dataframe with lag differences:

    #
    # Xlagged_diff = tmp - tmp.shift(1)
    # Xlagged_diff.columns = [f'diff_{i}' for i in Xlagged_diff.columns]
    # X_train_with_lags = pd.concat([tmp, Xlagged_diff], axis=1)
    #
    # Xlagged_diff_test = tmp_test - tmp_test.shift(1)
    # Xlagged_diff_test.columns = [f'diff_{i}' for i in Xlagged_diff_test.columns]
    # X_test_with_lags = pd.concat([tmp_test, Xlagged_diff_test], axis=1).fillna(method='bfill').fillna(method='ffill')
    tmp_X = pd.concat([data_dict[2016]['X_train'], data_dict[2017]['X_train'], data_dict[2018]['X_train']])
    tmp_Xtest = pd.concat([data_dict[2016]['X_test'], data_dict[2017]['X_test']])

    tmp_y = pd.concat([data_dict[2016]['y_train'], data_dict[2017]['y_train'], data_dict[2018]['y_train']])

    data_dict[2019] = {
        'X_train': tmp_X,
        'X_test': tmp_Xtest,
        "y_train": tmp_y,
        'X_train_ts': pd.concat(
            [data_dict[2016]['X_train_ts'], data_dict[2017]['X_train_ts'], data_dict[2018]['X_train_ts']]),
        'X_test_ts': pd.concat([data_dict[2016]['X_test_ts'], data_dict[2017]['X_test_ts']]),
        'X_train_tsclean': X_to_clean,
        'y_train_tsclean': y_to_clean,
        'mask': mask,
        'excl': df_excl,
        'X_train_rougher.output.recovery': tmp_X.filter(regex='rougher|hour|dayw', axis=1),
        # .drop(DROPCOLS_ROUGHER,axis = 1),
        'X_train_final.output.recovery': tmp_X  # .drop(DROPCOLS_FINAL,axis=1),
        # 'X_test_lagdiff': X_test_with_lags

        # 'X_train_pca': X_train_pca,
        # 'X_test_pca': X_test_pca
    }
    return data_dict


def calculate_interactions_diff_ratio(df, feature_list_rougher=['rougher.state.floatbank10_a_level',
                                                          'encod_val_rougher.input.feed_fe',
                                                          'value__quantile__q_0.95_p24_rougher.input.feed_fe',
                                                          'encod_val_rougher.input.feed_zn',
                                                          'rougher.input.feed_sol',
                                                          'value__quantile__q_0.95_p120_rougher.input.feed_pb',
                                                          'encod_val_rougher.input.feed_pb',
                                                          'rougher.input.floatbank10_xanthate',
                                                          'value__quantile__q_0.95_p120_rougher.input.feed_fe'],
                                feature_list_final=[
                                    "rougher.input.feed_fe",
                                    "primary_cleaner.input.copper_sulfate",
                                    "value__maximum_p6_rougher.input.feed_fe",
                                    "encod_val_primary_cleaner.input.feed_size",
                                    "deriv1_rougher.input.feed_zn",
                                    "value__minimum_p6_rougher.input.floatbank11_xanthate",
                                    "secondary_cleaner.state.floatbank2_a_air",
                                    "encod_dif_rougher.input.feed_pb",
                                    "value__quantile__q_0.95_p24_rougher.input.feed_fe",
                                    "encod_val_primary_cleaner.input.copper_sulfate"],eps = 0.1):
    df_sub_r = df[feature_list_rougher]
    df_sub_f = df[feature_list_final]
    print('Calculating DIFF interaction variables with rougher & final feature cols...')
    Ncol_r = len(df_sub_r.columns)
    Ncol_f = len(df_sub_f.columns)
    cols_r = df_sub_r.columns
    cols_f = df_sub_f.columns
    diff_r = np.empty((df_sub_r.shape[0], Ncol_r * (Ncol_r - 1) // 2))
    diff_f = np.empty((df_sub_f.shape[0], Ncol_f * (Ncol_f - 1) // 2))
    rat_r = np.empty((df_sub_r.shape[0], Ncol_r * (Ncol_r - 1) // 2))
    rat_f = np.empty((df_sub_f.shape[0], Ncol_f * (Ncol_f - 1) // 2))

    k = 0
    names_r = []
    names_f = []
    names_rat_f = []
    names_rat_r = []
    for i in range(Ncol_r):
        for j in range(i+1, Ncol_r):
            #print(k)
            name = f'{cols_r[i]}_diff_{cols_r[j]}'
            name_rat = f'{cols_r[i]}_rat_{cols_r[j]}'

            diff = df_sub_r[cols_r[i]] - df_sub_r[cols_r[j]]
            rat = df_sub_r[cols_r[i]] / (df_sub_r[cols_r[j]] + eps)
            diff_r[:, k] = diff
            rat_r[:, k] = rat
            k += 1

            names_r.append(name)
            names_rat_r.append(name_rat)
    k = 0
    for i in range(Ncol_f):
        for j in range(i+1, Ncol_f):
            name = f'{cols_f[i]}_diff_{cols_f[j]}'
            name_rat = f'{cols_f[i]}_rat_{cols_f[j]}'

            diff = df_sub_f[cols_f[i]] - df_sub_f[cols_f[j]]
            rat = df_sub_f[cols_f[i]] / (eps + df_sub_f[cols_f[j]])
            diff_f[:, k] = diff
            rat_f[:, k] = rat
            k += 1

            names_f.append(name)
            names_rat_f.append(name_rat)

    df_interaction_r = pd.DataFrame(data=diff_r, columns=names_r, index=df_sub_r.index)
    df_interaction_f = pd.DataFrame(data=diff_f, columns=names_f, index=df_sub_f.index)

    df_interaction_rat_r = pd.DataFrame(data=rat_r, columns=names_rat_r, index=df_sub_r.index)
    df_interaction_rat_f = pd.DataFrame(data=rat_f, columns=names_rat_f, index=df_sub_f.index)

    print('Concatenating interaction variables with rougher feature cols...')

    df_concat = pd.concat([df_interaction_r, df_interaction_f,df_interaction_rat_r,df_interaction_rat_f], axis=1)

    # print(feature_list_rougher)
    # print(feature_list_final)

    return df_concat


def calculate_interactions_poly(df, feature_list_rougher=['rougher.state.floatbank10_a_level',
                                                          'encod_val_rougher.input.feed_fe',
                                                          'value__quantile__q_0.95_p24_rougher.input.feed_fe',
                                                          'encod_val_rougher.input.feed_zn',
                                                          'rougher.input.feed_sol',
                                                          'value__quantile__q_0.95_p120_rougher.input.feed_pb',
                                                          'encod_val_rougher.input.feed_pb',
                                                          'rougher.input.floatbank10_xanthate',
                                                          'value__quantile__q_0.95_p120_rougher.input.feed_fe'],
                                feature_list_final=[
                                    "rougher.input.feed_fe",
                                    "primary_cleaner.input.copper_sulfate",
                                    "value__maximum_p6_rougher.input.feed_fe",
                                    "encod_val_primary_cleaner.input.feed_size",
                                    "deriv1_rougher.input.feed_zn",
                                    "value__minimum_p6_rougher.input.floatbank11_xanthate",
                                    "secondary_cleaner.state.floatbank2_a_air",
                                    "encod_dif_rougher.input.feed_pb",
                                    "value__quantile__q_0.95_p24_rougher.input.feed_fe",
                                    "encod_val_primary_cleaner.input.copper_sulfate"]):
    from sklearn.preprocessing import PolynomialFeatures

    poly_r = PolynomialFeatures(interaction_only=True, include_bias=False)
    poly_f = PolynomialFeatures(interaction_only=True, include_bias=False)
    df_sub_r = df[feature_list_rougher]
    df_sub_f = df[feature_list_final]
    print('Calculating interaction variables with rougher & final feature cols...')

    poly_f.fit(df_sub_f)
    poly_r.fit(df_sub_r)

    names_r = poly_r.get_feature_names(df_sub_r.columns)
    names_r = [x.replace(" ", "_poly_") for x in names_r]
    names_f = poly_f.get_feature_names(df_sub_f.columns)
    names_f = [x.replace(" ", "_poly_") for x in names_f]

    print('Transforming interaction variables with rougher feature cols...')

    df_interaction_r = pd.DataFrame(data=poly_r.transform(df_sub_r), columns=names_r, index=df_sub_r.index)
    df_interaction_f = pd.DataFrame(data=poly_f.transform(df_sub_f), columns=names_f, index=df_sub_f.index)
    print('Concatenating interaction variables with rougher feature cols...')

    df_concat = pd.concat([df_interaction_r, df_interaction_f], axis=1)

    # print(feature_list_rougher)
    # print(feature_list_final)

    return df_concat


def create_meta_features(df, lag=6, eps=0.1):
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
                    "primary_cleaner.input.xanthate"]

    for c in cols_to_calc:
        colname_meta = f'{c}_normalized_lag_{lag}'
        colname_min = f'value__minimum_p{lag}_{c}'
        colname_max = f'value__maximum_p{lag}_{c}'
        df[colname_meta] = (df[c] - df[colname_min]) / (df[colname_max] - df[colname_min] + eps)

    return df


def calculate_lagdiffs_features(df=None, base_lag=0, step_lag=1,
                                cols_to_diff=('rougher.input.floatbank10_xanthate', 'primary_cleaner.input.xanthate',
                                              'rougher.input.feed_pb',
                                              'primary_cleaner.input.depressant', 'encod_val_rougher.input.feed_zn',
                                              'rougher.input.floatbank11_xanthate',
                                              'primary_cleaner.state.floatbank8_d_level'), relative=False):
    if relative:
        df_lagged_diff = (df.shift(base_lag) - df.shift(step_lag)) / (df.shift(base_lag) + df.shift(step_lag))

        df_lagged_diff = df_lagged_diff.loc[:, cols_to_diff].fillna(method='bfill')
    else:
        df_lagged_diff = (df.shift(base_lag) - df.shift(step_lag)).loc[:, cols_to_diff].fillna(method='bfill')

    prefix = 'rel' if relative else 'diff'
    df_lagged_diff.columns = [f'{prefix}_{i}_s{step_lag}b{base_lag}' for i in df_lagged_diff.columns]

    return df_lagged_diff


def calculate_derivative_features(df=pd.DataFrame(), order=2, cols_to_diff=['A', 'B', 'C']):
    df_deriv = df
    for i in range(order):
        df_deriv = df_deriv.diff(1)
    df_deriv = df_deriv.loc[:, cols_to_diff]
    df_deriv.columns = [f'deriv{order}_{i}' for i in df_deriv.columns]

    return df_deriv.fillna(method='bfill')


def encode_oof(model, X_base=None, y=None, n_folds=4, col_id='ABC'):
    cv = KFold(n_folds, shuffle=False, random_state=42)
    preds = cross_val_predict(model, X=X_base, y=y, cv=cv, n_jobs=n_folds, method='predict')
    diffs = y - preds
    rels = (y - preds) / (np.abs(y) + 0.5)
    df_res = pd.DataFrame({f'encod_val_{col_id}': preds, f'encod_rel_{col_id}': rels, f'encod_dif_{col_id}': diffs},
                          index=X_base.index)
    return df_res


def encode_imp_features(feats_rougher=('rougher.input.feed_fe', 'rougher.input.feed_pb', 'rougher.input.feed_zn'),
                        feats_final=('primary_cleaner.input.xanthate',
                                     'primary_cleaner.state.floatbank8_a_level',
                                     'primary_cleaner.input.feed_size', 'primary_cleaner.input.depressant',
                                     'primary_cleaner.input.copper_sulfate'), df=pd.DataFrame()):
    params_xg = {'max_depth': 3, 'gamma': '0.544', 'colsample_bytree': '0.684', 'subsample': '0.932'}

    model_xgb = xgb.XGBRegressor(learning_rate=0.05,
                                 n_estimators=200,
                                 random_state=7, nthread=-1, **params_xg)

    X = df.copy().filter(regex='rougher', axis=1)
    # Encode rougher features first:
    df_encods = []
    for f in feats_rougher:
        tmp = encode_oof(model_xgb, X.drop(f, axis=1), X[f], col_id=f)
        df_encods.append(tmp)
    # Encode final features
    X = df.copy()
    for f in feats_final:
        tmp = encode_oof(model_xgb, X.drop(f, axis=1), X[f], col_id=f)
        df_encods.append(tmp)
    encoded_df = pd.concat(df_encods, axis=1)
    return encoded_df


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
    # "diff_rougher.input.floatbank11_copper_sulfate",
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
    "primary_cleaner.state.floatbank8_d_level",
    "rougher.input.floatbank10_copper_sulfate",
    "encod_val_rougher.input.feed_pb",
    "primary_cleaner.state.floatbank8_c_level",
    "rougher.input.feed_sol",
    "primary_cleaner.state.floatbank8_c_air",
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
    "rougher.input.feed_size",
    "encod_val_rougher.input.feed_fe",
    "rougher.input.feed_rate",
    "encod_rel_rougher.input.feed_fe",
    "rougher.state.floatbank10_d_level",
    "encod_dif_rougher.input.feed_fe"
])

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    data_dict = split_data(root=project_dir)
    with open(f'{project_dir}/data/processed/data_dict_all.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    print('Done splitting!')
