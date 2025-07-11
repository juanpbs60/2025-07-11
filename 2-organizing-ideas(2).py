#!/usr/bin/env python
# coding: utf-8

########################  FRAMEWORKS  ########################

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

########################  FILE DOWNLOADING  ########################

DATASETS_DIR = 'datasets/' 
RETRIEVED_DATA = 'raw-data.csv'

def data_retrieval(url):
     
    # Loading data from specific url
    data = pd.read_csv(url)
    
    # Uncovering missing data
    data.replace('?', np.nan, inplace=True)
    data['age'] = data['age'].astype('float')
    data['fare'] = data['fare'].astype('float')
    
    # helper function 1
    def get_first_cabin(row):
        try:
            return row.split()[0]
        except:
            return np.nan
    
    # helper function 2
    def get_title(passenger):
        line = passenger
        if re.search('Mrs', line):
            return 'Mrs'
        elif re.search('Mr', line):
            return 'Mr'
        elif re.search('Miss', line):
            return 'Miss'
        elif re.search('Master', line):
            return 'Master'
        else:
            return 'Other'
    
    # Keep only one cabin | Extract the title from 'name'
    data['cabin'] = data['cabin'].apply(get_first_cabin)
    data['title'] = data['name'].apply(get_title)
    
    # Droping irrelevant columns
    DROP_COLS = ['boat','body','home.dest','ticket','name']
    data.drop(DROP_COLS, 1, inplace=True)
    
    data.to_csv(DATASETS_DIR + RETRIEVED_DATA, index=False)
    
    return print('Data stored in {}'.format(DATASETS_DIR + RETRIEVED_DATA))

URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
data_retrieval(URL)

df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)
df.shape

df.sample(5)

########################  TRAIN-TEST SPLIT  ########################

SEED_SPLIT = 404

X_train, X_test, y_train, y_test = train_test_split(
                                                        df.drop('survived', axis=1),
                                                        df['survived'],
                                                        test_size=0.2,
                                                        random_state=SEED_SPLIT
                                                   )

TRAIN_DATA_FILE = DATASETS_DIR + 'train.csv'
TEST_DATA_FILE  = DATASETS_DIR + 'test.csv'

X_train.to_csv(TRAIN_DATA_FILE, index=False)
X_test.to_csv(TEST_DATA_FILE, index=False)

X_train.shape, X_test.shape

target = 'survived'
num_vars = [col for col in X_train.columns if X_train[col].dtype != object and col != target]
cat_vars = [col for col in X_train.columns if X_train[col].dtype == object]

# Validation step
len(num_vars) + len(cat_vars) + 1 == df.shape[1]

########################  FEATURE ENGINEERING  ########################

# ### 3.1. Without persisting information

# **Numerical variables**
# 
# - Create missing value indicator: only for numeric variables

def missing_indicator(data, col_name):
    data[col_name+'_nan'] = data[col_name].isnull().astype(int)
    return None

for var in num_vars:
    missing_indicator(X_train, var)
    missing_indicator(X_test, var)

X_train.head(2)

# **Categorical variables**
# 
# - Keep only the letter in cabin
# - Fill NaN with label "missing"

def extract_letter_from_cabin(x):
    if type(x)==str:    
        return ''.join(re.findall("[a-zA-Z]+", x))  
    else: 
        return x

X_train['cabin'] = X_train['cabin'].apply(extract_letter_from_cabin)    
X_test['cabin'] = X_test['cabin'].apply(extract_letter_from_cabin)    

X_train['cabin'].unique(), X_test['cabin'].unique()

X_train[cat_vars] = X_train[cat_vars].fillna('missing')
X_test[cat_vars]  = X_test[cat_vars].fillna('missing')

# ### 3.2. With persisting information

# **Numerical variables**
# 
# - Fill NaN with median

imp_median = SimpleImputer(strategy='median')
imp_median.fit(X_train[num_vars])

imp_median.statistics_

X_train[num_vars] = imp_median.transform(X_train[num_vars])
X_test[num_vars]  = imp_median.transform(X_test[num_vars])

# **Categorical variables**
# 
# - Remove rare labels
# - One hot encoding
# - Fix one-hot-encoded features not in test set

def find_rare_labels(data, col, perc):
    data = data.copy()
    tmp = data.groupby(col)[col].count() / data.shape[0]
    return tmp[tmp < perc].index

rare_labels_ = {}
for col in cat_vars:
    rare_labels_[col] = find_rare_labels(X_train, col, 0.05)
    
for col in cat_vars:
    X_train[col] = np.where(X_train[col].isin(rare_labels_[col]), 'Rare', X_train[col])
    X_test[col]  = np.where(X_test[col].isin(rare_labels_[col]), 'Rare', X_test[col])

X_train[cat_vars[1]].unique()

X_train = pd.concat([X_train, pd.get_dummies(X_train[cat_vars], drop_first=True)], 1)
X_test  = pd.concat([X_test, pd.get_dummies(X_test[cat_vars], drop_first=True)], 1)

X_train.drop(cat_vars, 1, inplace=True)
X_test.drop(cat_vars, 1, inplace=True)

# Validation step
set(X_train.columns).difference(set(X_test.columns))

for col in list(set(X_train.columns).difference(set(X_test.columns))):
    X_test[col] = 0

# **Aligning columns of X_train and X_test**

ordered_vars = [col for col in X_train.columns]

X_train = X_train[ordered_vars]
X_test  = X_test[ordered_vars]

########################  SCALING  ########################

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

########################  ML TRAINING  ########################
SEED_MODEL = 404

model = LogisticRegression(C=0.0005, class_weight='balanced', random_state=SEED_MODEL)
model.fit(X_train, y_train)

for s,t in zip(['train','test'],[(X_train, y_train),(X_test,y_test)]):
    x,y = t[0], t[1]
    class_pred = model.predict(x)
    proba_pred = model.predict_proba(x)[:,1]
    print('{} roc-auc : {}'.format(s, roc_auc_score(y, proba_pred)))
    print('{} accuracy: {}'.format(s, accuracy_score(y, class_pred)))
    print()

tmp = pd.DataFrame(X_test, columns=ordered_vars)
tmp['y_true'] = np.array(y_test)
tmp['y_pred'] = model.predict(X_test)
tmp['proba_pred'] = model.predict_proba(X_test)[:,1]

tmp.head(10)
