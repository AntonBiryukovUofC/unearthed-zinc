import logging
from pathlib import Path
import pandas as pd
import pickle


def split_data(root):
    # TSFRESH

    df_tsfresh = pd.read_pickle(f'{root}/data/processed/train_test_tsfresh.pkl').reset_index(level=0)
    df_flat = df_tsfresh.pivot_table(index='id', columns=['level_0'])
    df_flat.index.names = ['date']
    df_flat.columns = ['_'.join(col).strip() for col in df_flat.columns.values]

    df_all = pd.read_pickle(f'{root}/data/processed/train_test_raw.pkl')
    df_pca = pd.read_pickle(f'{root}/data/processed/train_test_PCA.pkl')

    data_dict = {}
    years = [2016, 2017,2018]
    tgts = ['final.output.recovery', 'rougher.output.recovery']
    df_all = df_all.reset_index('date')

    for y in years:
        inds = df_all['date'].dt.year == y
        X_train = df_all[inds].loc['train',].drop(tgts, axis=1).set_index('date', append=True).reset_index(level=0,
                                                                                                           drop=True)
        X_train_ts = X_train.join(df_flat, how='inner')
        X_train_pca = df_pca.loc[('train',),:].loc[X_train.index]

        y_train = df_all[inds].loc['train',][tgts + ['date']].set_index('date', append=True).reset_index(level=0,
                                                                                                         drop=True)
        if y != 2018:
            X_test = df_all[inds].loc['test',].drop(tgts, axis=1).set_index('date', append=True).reset_index(level=0,
                                                                                                         drop=True)

            X_test_ts = X_test.join(df_flat, how='inner')
            X_test_pca = df_pca.loc[('test',),].loc[X_test.index]
        else:
            X_test = None
            X_test_pca = None
        data_dict[y] = {'X_train': X_train, 'X_test': X_test, "y_train": y_train, 'X_train_ts': X_train_ts,
                        'X_test_ts': X_test_ts, 'X_train_pca': X_train_pca, 'X_test_pca': X_test_pca}

    data_dict[2019] =  {'X_train': pd.concat([data_dict[2016]['X_train'],data_dict[2017]['X_train'],data_dict[2018]['X_train']]),
                        'X_test': pd.concat([data_dict[2016]['X_test'],data_dict[2017]['X_test'],data_dict[2018]['X_test']]),
                        "y_train": pd.concat([data_dict[2016]['y_train'],data_dict[2017]['y_train'],data_dict[2018]['y_train']]),
                        'X_train_ts': pd.concat([data_dict[2016]['X_train_ts'],data_dict[2017]['X_train_ts'],data_dict[2018]['X_train_ts']]),
                        'X_test_ts': pd.concat([data_dict[2016]['X_test_ts'],data_dict[2017]['X_test_ts'],data_dict[2018]['X_test_ts']])
                        #'X_train_pca': X_train_pca,
                        #'X_test_pca': X_test_pca
                         }
    return data_dict


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