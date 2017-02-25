# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd 
import xgboost as xgb
import os
from scipy.stats import norm, lognorm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

os.chdir('/Users/weileizhang/Documents/programming/python_code/kaggle/Allstate')
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = [x for x in train.columns if x not in ['id','loss']]

cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]
num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]



train['log_loss'] = np.log(train['loss'])
ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train[features], test[features])).reset_index(drop=True)
for c in range(len(cat_features)):
    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes

train_x = train_test.iloc[:ntrain,:]
test_x = train_test.iloc[ntrain:,:]

xgdmat = xgb.DMatrix(train_x, train['log_loss']) # Create our DMatrix to make XGBoost more efficient

params = {'eta': 0.01, 'seed':7, 'subsample': 1, 'colsample_bytree': 1, 
             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':100, 'silent':0} 

num_rounds = 1000
bst = xgb.train(params, xgdmat, num_boost_round = num_rounds)

test_xgb=xgb.DMatrix(test_x)
submission = pd.read_csv("sample_submission.csv")
submission.iloc[:, 1] = np.exp(bst.predict(test_xgb))
submission.to_csv('xgb_starter.sub.csv', index=None)
