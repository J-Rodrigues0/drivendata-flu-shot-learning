#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

plt.rcParams['figure.figsize'] = [8.0, 8.0]
plt.rcParams['figure.dpi'] = 140


# # Loading data

# In[2]:


X_train_full_seas = pd.read_csv('interim_data\\preprocessed_train_features_seas.csv', index_col = 'respondent_id')
X_train_full_h1n1 = pd.read_csv('interim_data\\preprocessed_train_features_h1n1.csv', index_col = 'respondent_id')

y_train_full = pd.read_csv('input_data\\training_set_labels.csv', index_col = 'respondent_id')


# In[3]:


train_df_seas = X_train_full_seas.join(y_train_full)
train_df_h1n1 = X_train_full_h1n1.join(y_train_full)


# # Loading models

# In[4]:


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier


# # Hyperparameter Tuning

# ## Importing Optuna

# In[5]:


import optuna
from optuna import Trial, visualization


# ## Making validation folds

# In[6]:


from sklearn.model_selection import KFold

N_folds = 5
kf = KFold(n_splits=N_folds, random_state=42, shuffle=True)

for i, (trn, val) in enumerate(kf.split(train_df_seas)):
    train_df_seas.loc[val, 'kfold'] = i
    train_df_h1n1.loc[val, 'kfold'] = i
    
train_df_seas['kfold'] = train_df_seas['kfold'].astype(int)
train_df_h1n1['kfold'] = train_df_h1n1['kfold'].astype(int)


# ## XGBoost Tuning

# ### XGBoost - H1N1 Vaccine

# In[7]:


feature_cols = [col for col in X_train_full_h1n1.columns.tolist() if col not in ['h1n1_vaccine', 'seasonal_vaccine', 'kfold']]
target_cols = ['h1n1_vaccine']


# In[8]:


def objective_xgb_h1n1(trial):
    roc = 0
    for fold in range(5):
        trn_idx = train_df_h1n1['kfold'] != fold
        val_idx = train_df_h1n1['kfold'] == fold
        trn = train_df_h1n1.loc[trn_idx, :]
        val = train_df_h1n1.loc[val_idx, :]

        x_tr, y_tr = trn[feature_cols].values, trn[target_cols].values
        x_val, y_val = val[feature_cols].values, val[target_cols].values
        
        model, log = fit_xgb(trial, x_tr, y_tr, x_val, y_val)
        roc += log['valid roc']/5
        
    return roc

def objective_xgb_seas(trial):
    roc = 0
    for fold in range(5):
        trn_idx = train_df_seas['kfold'] != fold
        val_idx = train_df_seas['kfold'] == fold
        trn = train_df_seas.loc[trn_idx, :]
        val = train_df_seas.loc[val_idx, :]

        x_tr, y_tr = trn[feature_cols].values, trn[target_cols].values
        x_val, y_val = val[feature_cols].values, val[target_cols].values
        
        model, log = fit_xgb(trial, x_tr, y_tr, x_val, y_val)
        roc += log['valid roc']/5
        
    return roc


# In[9]:


from sklearn.metrics import roc_auc_score

def fit_xgb(trial, x_train, y_train, x_val, y_val):
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [80, 85, 90, 95, 100, 105, 110, 115, 120]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.025, 0.03, 0.04, 0.05, 0.75, 0.1]),
        "subsample": trial.suggest_discrete_uniform("subsample", 0.4,1,0.1),
        "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.6,1,0.1),
        "max_depth": trial.suggest_categorical("max_depth",[4,5,6,7,8,9,10]),
        "min_child_weight": trial.suggest_int("min_child_weight",1,5),
        "gamma": 0,
        "base_score": 0.5,
        "random_state": 42,
        "use_label_encoder": False,
        "objective": 'reg:logistic',
        "tree_method": 'exact'
    }
    
    model = XGBClassifier(**params)
    model.fit(x_train, y_train.reshape(-1,))

    y_pred_tr = model.predict_proba(x_train)[:, 1]
    y_pred_val = model.predict_proba(x_val)[:, 1]

    log = {
        "train roc": roc_auc_score(y_train, y_pred_tr),
        "valid roc": roc_auc_score(y_val, y_pred_val)
    }
    
    return model, log


# In[10]:


XGB_study_H1N1 = optuna.create_study(direction="maximize", study_name='XGBoost H1N1 Vaccine optimization')
XGB_study_H1N1.optimize(objective_xgb_h1n1, n_trials=10)


# In[11]:


XGB_study_H1N1.best_params


# In[12]:


import pickle
from pathlib import Path

root = Path(".")

with open(root / "models\\XGB_H1N1_best_params.pkl","wb") as f:
    pickle.dump(XGB_study_H1N1.best_params, f)


# ### XGBoost - Seasonal Vaccine

# In[13]:


feature_cols = [col for col in X_train_full_seas.columns.tolist() if col not in ['h1n1_vaccine', 'seasonal_vaccine', 'kfold']]
target_cols = ['seasonal_vaccine']


# In[14]:


XGB_study_SEAS = optuna.create_study(direction="maximize", study_name='XGBoost Seasonal Vaccine optimization')
XGB_study_SEAS.optimize(objective_xgb_seas, n_trials=10)


# In[15]:


XGB_study_SEAS.best_params


# In[16]:


with open(root / "models\\XGB_SEAS_best_params.pkl","wb") as f:
    pickle.dump(XGB_study_SEAS.best_params, f)


# ## LightGBM

# ### LightGBM - H1N1 Vaccine

# In[17]:


feature_cols = [col for col in X_train_full_h1n1.columns.tolist() if col not in ['h1n1_vaccine', 'seasonal_vaccine', 'kfold']]
target_cols = ['h1n1_vaccine']


# In[18]:


def objective_lgbm_h1n1(trial):
    roc = 0
    for fold in range(5):
        trn_idx = train_df_h1n1['kfold'] != fold
        val_idx = train_df_h1n1['kfold'] == fold
        trn = train_df_h1n1.loc[trn_idx, :]
        val = train_df_h1n1.loc[val_idx, :]

        x_tr, y_tr = trn[feature_cols].values, trn[target_cols].values
        x_val, y_val = val[feature_cols].values, val[target_cols].values
        
        model, log = fit_lgbm(trial, x_tr, y_tr, x_val, y_val)
        roc += log['valid roc']/5
        
    return roc

def objective_lgbm_seas(trial):
    roc = 0
    for fold in range(5):
        trn_idx = train_df_seas['kfold'] != fold
        val_idx = train_df_seas['kfold'] == fold
        trn = train_df_seas.loc[trn_idx, :]
        val = train_df_seas.loc[val_idx, :]

        x_tr, y_tr = trn[feature_cols].values, trn[target_cols].values
        x_val, y_val = val[feature_cols].values, val[target_cols].values
        
        model, log = fit_lgbm(trial, x_tr, y_tr, x_val, y_val)
        roc += log['valid roc']/5
        
    return roc


# In[19]:


def fit_lgbm(trial, x_train, y_train, x_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 60, 150, 10),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.025, 0.05, 0.75, 0.1]),
        "subsample": trial.suggest_discrete_uniform("subsample", 0.8,1,0.05),
        "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.6,0.9,0.05),
        "min_child_samples": trial.suggest_int("min_child_samples",18,30),
        "min_child_weight": trial.suggest_discrete_uniform("min_child_weight",0.0005,0.0015,0.0005),
        "max_depth": -1,
        "random_state": 42,
        "silent":True
    }
    
    model = LGBMClassifier(**params)
    model.fit(x_train, y_train.reshape(-1,))

    y_pred_tr = model.predict_proba(x_train)[:, 1]
    y_pred_val = model.predict_proba(x_val)[:, 1]

    log = {
        "train roc": roc_auc_score(y_train, y_pred_tr),
        "valid roc": roc_auc_score(y_val, y_pred_val)
    }
    
    return model, log


# In[20]:


LGBM_study_H1N1 = optuna.create_study(direction="maximize", study_name='LightGBM H1N1 Vaccine optimization')
LGBM_study_H1N1.optimize(objective_lgbm_h1n1, n_trials=10)


# In[21]:


LGBM_study_H1N1.best_params


# In[22]:


with open(root / "models\\LGBM_H1N1_best_params.pkl","wb") as f:
    pickle.dump(LGBM_study_H1N1.best_params, f)


# ### LightGBM - Seasonal Vaccine

# In[23]:


feature_cols = [col for col in X_train_full_seas.columns.tolist() if col not in ['h1n1_vaccine', 'seasonal_vaccine', 'kfold']]
target_cols = ['seasonal_vaccine']


# In[24]:


LGBM_study_SEAS = optuna.create_study(direction="maximize", study_name='LightGBM Seasonal Vaccine optimization')
LGBM_study_SEAS.optimize(objective_lgbm_seas, n_trials=10)


# In[25]:


LGBM_study_SEAS.best_params


# In[26]:


with open(root / "models\\LGBM_SEAS_best_params.pkl","wb") as f:
    pickle.dump(LGBM_study_SEAS.best_params, f)

