
import logging
from pathlib import Path
import pandas as pd
import pickle


def split_data(root):
    df_tsfresh_feats = pd.read_pickle(f'{root}/data/processed/train_test_tsfresh.pkl')
    df_all = pd.read_pickle(f'{root}/data/processed/train_test_raw.pkl')
    data_dict = {}
    years = [2016,2017]
    tgts = ['final.output.recovery','rougher.output.recovery']
    df_all = df_all.reset_index('date')
    for y in years:
        inds = df_all['date'].dt.year <= y
        X_train = df_all[inds].loc['train',].drop(tgts,axis = 1).set_index('date',append=True)
        y_train = df_all[inds].loc['train',][tgts+['date']].set_index('date',append=True)
        X_test = df_all[inds].loc['test',].drop(tgts,axis = 1).set_index('date',append=True)
        data_dict[y] = {'X_train':X_train,'X_test':X_test,"y_train":y_train}
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
    with open(f'{project_dir}/data/processed/data_dict_all.pkl','wb') as f:
        pickle.dump(data_dict,f)
