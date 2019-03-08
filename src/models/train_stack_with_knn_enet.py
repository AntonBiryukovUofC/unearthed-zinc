import sys

import pandas as pd

sys.path.insert(0, '..')
from src.models.model import stacks, stackerTraining
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from src.models.model import level0_models_rougher, level0_models_final

# Load the data for the rougher.output.recovery target
tgt = 'rougher.output.recovery'
data_dict = pd.read_pickle(f'../data/processed/data_dict_all.pkl')
year = 2019  # all_data means I used the fake 2019 dataset, that has all years in a training set
X = data_dict[year]['X_train_tsclean']
y = data_dict[year]['y_train']
X_test = data_dict[year]['X_test_ts']
mask = data_dict[year]['mask']
print(X.shape)
print(f'1) X shape: {X.shape},y: {y.shape}')
X = X[mask]
y = y[mask][tgt]
print(f'2) Train shape: {X.shape}')
X_filt = X.filter(regex="rougher|hour|dayw", axis=1) # Keep upstream variables here only
X = X_filt
train_df = pd.concat([X, y], axis=1)

# Set up folding strategy
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=False, random_state=156)
# A dictionary of base models
scaler = make_pipeline(QuantileTransformer(output_distribution='normal'), PCA(whiten=True))
# Train the stack of level0 models for Rougher Target
level0_trainFeatures_rougher, level0_testFeatures_rougher = stacks(level0_models_rougher, kf, X.columns, tgt, train_df,
                                                                   X_test)
# Train the stackers:
# A dictionary of level 1 model to train on base model predictions
stacker_r = {
    'Enet': ElasticNet(alpha=0.001),
    'Lasso': Lasso(alpha=0.005, random_state=1, max_iter=2000),
    'positive_Lasso': Lasso(alpha=0.0001, precompute=True, max_iter=1000, positive=True, random_state=9999,
                            selection='random'),
}

stackerTraining(stacker_r, kf, level0_trainFeatures_rougher, level0_testFeatures_rougher, train_df,tgt)
level1_model = stacker_r['Lasso'].fit(level0_trainFeatures_rougher, train_df[tgt])
level1_test_pred_rougher = level1_model.predict(level0_testFeatures_rougher)
rougher_meta_train = level0_trainFeatures_rougher
rougher_meta_test = level0_testFeatures_rougher

# Repeat for the final.output.target
year = 2019
tgt = 'final.output.recovery'
X = data_dict[year]['X_train_tsclean']
y = data_dict[year]['y_train_tsclean']
X_test = data_dict[year]['X_test_ts']
mask = data_dict[year]['mask']
print(X.shape)
print(f'1) X shape: {X.shape},y: {y.shape}')
X = X[mask]
y = y[mask][tgt]
train_df = pd.concat([X,y],axis= 1)
print(f'1) X shape: {X.shape},y: {y.shape}')
train_df.head()

level0_trainFeatures_final, level0_testFeatures_final = stacks(level0_models_final, kf, X.columns, tgt, train_df, X_test)
stacker_f = {
           'Enet': ElasticNet(alpha = 0.001),
          'Lasso': Lasso(alpha =0.005, random_state=1,max_iter = 2000)
}
stackerTraining(stacker_f, kf, level0_trainFeatures_final, level0_testFeatures_final, train_df,target = tgt)
level1_model = stacker_f['Lasso'].fit(level0_trainFeatures_final,train_df[tgt])
level1_test_pred_final = level1_model.predict(level0_testFeatures_final)


# Make a submission:
preds = pd.DataFrame(data = {'date':X_test.index,'rougher.output.recovery':level1_test_pred_rougher, 'final.output.recovery':level1_test_pred_final}).set_index('date')
stacked_preds_sub = preds
stacked_preds_sub = stacked_preds_sub.reset_index()
stacked_preds_sub['date'] = stacked_preds_sub['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
stacked_preds_sub.set_index('date',inplace=True)
stacked_preds_sub.drop_duplicates(inplace=True)
stacked_preds_sub.to_csv('../results/stacked_sub_lgb_lasso_base_alldata_tsclean.csv')

# Make a submission with averaged preds instead of a meta-estimator:
preds_av = pd.DataFrame(data = {'date':X_test.index,'rougher.output.recovery':level0_testFeatures_rougher.mean(axis=1), 'final.output.recovery':level0_testFeatures_final.mean(axis=1)})
preds_av['date'] = preds_av['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
preds_av.set_index('date',inplace=True)
preds_av.to_csv('../results/stacked_sub_lgb_lasso_base_alldata_tsclean_mod_averaged.csv')