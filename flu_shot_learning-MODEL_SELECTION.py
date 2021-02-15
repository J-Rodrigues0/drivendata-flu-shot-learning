#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

root = Path(".")

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


# # Loading test models

# In[4]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


# In[5]:


LogReg_class_h1n1 = LogisticRegressionCV(max_iter=1000, n_jobs=-1, random_state=42)
LogReg_class_seas = LogisticRegressionCV(max_iter=1000, n_jobs=-1, random_state=42)


# In[6]:


KNN_class_h1n1 = KNeighborsClassifier(n_neighbors=200, weights='distance', n_jobs=-1)
KNN_class_seas = KNeighborsClassifier(n_neighbors=200, weights='distance', n_jobs=-1)


# In[7]:


MLP_class_h1n1 = MLPClassifier(learning_rate='adaptive', alpha=0.01, max_iter=1000, early_stopping=True, tol=1e-6, random_state=42)
MLP_class_seas = MLPClassifier(learning_rate='adaptive', alpha=0.01, max_iter=1000, early_stopping=True, tol=1e-6, random_state=42)


# In[8]:


GaussNB_class_h1n1 = GaussianNB(var_smoothing=1e-1)
GaussNB_class_seas = GaussianNB(var_smoothing=1e-1)


# In[9]:


from xgboost import XGBClassifier

with open(root / "models\\XGB_H1N1_best_params.pkl","rb") as f:
    XGB_params_h1n1 = pickle.load(f)

XGB_class_h1n1 = XGBClassifier(**XGB_params_h1n1, objective='reg:logistic', random_state=42, use_label_encoder=False)

with open(root / "models\\XGB_SEAS_best_params.pkl","rb") as f:
    XGB_params_seas = pickle.load(f)

XGB_class_seas = XGBClassifier(**XGB_params_seas, objective='reg:logistic', random_state=42, use_label_encoder=False)


# In[10]:


from lightgbm import LGBMClassifier

with open(root / "models\\LGBM_H1N1_best_params.pkl","rb") as f:
    LGBM_params_h1n1 = pickle.load(f)

LGBM_class_h1n1 = LGBMClassifier(**LGBM_params_h1n1)

with open(root / "models\\LGBM_SEAS_best_params.pkl","rb") as f:
    LGBM_params_seas = pickle.load(f)

LGBM_class_seas = LGBMClassifier(**LGBM_params_seas)


# In[11]:


from catboost import CatBoostClassifier

CAT_params_h1n1 = {}

CAT_class_h1n1 = CatBoostClassifier(**CAT_params_h1n1)

CAT_params_seas = {}

CAT_class_seas = CatBoostClassifier(**CAT_params_seas)


# In[13]:


from sklearn.ensemble import StackingClassifier

STACK_class_h1n1 = StackingClassifier(estimators=[('Logist', LogReg_class_h1n1),
                                                  ('KNN', KNN_class_h1n1),
                                                  ('MLP', MLP_class_h1n1),
                                                  ('GaussNB', GaussNB_class_h1n1),
                                                  ('XGBoost', XGB_class_h1n1),
                                                  ('LightGBM', LGBM_class_h1n1),
                                                  ('CatBoost', CAT_class_h1n1)],
                                      n_jobs=-1,
                                      verbose=3)

STACK_class_seas = StackingClassifier(estimators=[('Logist', LogReg_class_seas),
                                                  ('KNN', KNN_class_seas),
                                                  ('MLP', MLP_class_seas),
                                                  ('GaussNB', GaussNB_class_seas),
                                                  ('XGBoost', XGB_class_seas),
                                                  ('LightGBM', LGBM_class_seas),
                                                  ('CatBoost', CAT_class_seas)],
                                      n_jobs=-1,
                                      verbose=3)


# In[14]:


models_h1n1 = {'Logist': LogReg_class_h1n1,
               'KNN': KNN_class_h1n1,
               'MLP': MLP_class_h1n1,
               'GaussNB': GaussNB_class_h1n1,
               'XGB': XGB_class_h1n1,
               'LGBM': LGBM_class_h1n1,
               'CAT': CAT_class_h1n1,
               'STACK': STACK_class_h1n1}


# In[15]:


models_seas = {'Logist': LogReg_class_seas,
               'KNN': KNN_class_seas,
               'MLP': MLP_class_seas,
               'GaussNB': GaussNB_class_seas,
               'XGB': XGB_class_seas,
               'LGBM': LGBM_class_seas,
               'CAT': CAT_class_seas,
               'STACK': STACK_class_seas}


# # Cross-validating models

# In[16]:


from sklearn.model_selection import KFold

N_folds = 5
kf = KFold(n_splits=N_folds, random_state=42, shuffle=True)

for i, (trn, val) in enumerate(kf.split(train_df_seas)):
    train_df_seas.loc[val, 'kfold'] = i
    train_df_h1n1.loc[val, 'kfold'] = i
    
train_df_seas['kfold'] = train_df_seas['kfold'].astype(int)
train_df_h1n1['kfold'] = train_df_h1n1['kfold'].astype(int)


# In[17]:


from sklearn.metrics import roc_auc_score, roc_curve


# In[18]:


def cross_validate_model_seas(model_name, model_dict, mean_roc_dict, fit_params = {}):
    model = model_dict[model_name]
    print(model_name)
    mean_roc = 0
    for fold in range(N_folds):
        print(f'Fold: {fold}')
        train_idx = train_df_seas['kfold'] != fold
        valid_idx = train_df_seas['kfold'] == fold
        train = train_df_seas.loc[train_idx, :]
        valid = train_df_seas.loc[valid_idx, :]
        
        x_train, y_train = train[feature_cols].values, train[target_cols].values
        x_valid, y_valid = valid[feature_cols].values, valid[target_cols].values
        
        model.fit(x_train, y_train.reshape(-1,), **fit_params)
        
        try:
            y_pred_valid = model.predict_proba(x_valid)[:, 1]
        except:
            y_pred_valid = model.predict(x_valid)
            
        mean_roc += roc_auc_score(y_valid, y_pred_valid)/5
        
    mean_roc_dict[model_name] = [model, mean_roc]


# In[19]:


def cross_validate_model_h1n1(model_name, model_dict, mean_roc_dict, fit_params = {}):
    model = model_dict[model_name]
    print(model_name)
    mean_roc = 0
    for fold in range(N_folds):
        print(f'Fold: {fold}')
        train_idx = train_df_h1n1['kfold'] != fold
        valid_idx = train_df_h1n1['kfold'] == fold
        train = train_df_h1n1.loc[train_idx, :]
        valid = train_df_h1n1.loc[valid_idx, :]
        
        x_train, y_train = train[feature_cols].values, train[target_cols].values
        x_valid, y_valid = valid[feature_cols].values, valid[target_cols].values
        
        model.fit(x_train, y_train.reshape(-1,), **fit_params)
        
        try:
            y_pred_valid = model.predict_proba(x_valid)[:, 1]
        except:
            y_pred_valid = model.predict(x_valid)
            
        mean_roc += roc_auc_score(y_valid, y_pred_valid)/5
        
    mean_roc_dict[model_name] = [model, mean_roc]


# ## H1N1 Label

# In[20]:


mean_roc_dict_h1n1 = {}


# In[21]:


feature_cols = [col for col in X_train_full_h1n1.columns.tolist() if col not in ['h1n1_vaccine', 'seasonal_vaccine']]
target_cols = ['h1n1_vaccine']


# In[22]:


cross_validate_model_h1n1('Logist', models_h1n1, mean_roc_dict_h1n1)


# In[23]:


cross_validate_model_h1n1('KNN', models_h1n1, mean_roc_dict_h1n1)


# In[24]:


cross_validate_model_h1n1('MLP', models_h1n1, mean_roc_dict_h1n1)


# In[25]:


cross_validate_model_h1n1('GaussNB', models_h1n1, mean_roc_dict_h1n1)


# In[26]:


cross_validate_model_h1n1('XGB', models_h1n1, mean_roc_dict_h1n1)


# In[27]:


cross_validate_model_h1n1('LGBM', models_h1n1, mean_roc_dict_h1n1)


# In[28]:


cross_validate_model_h1n1('CAT', models_h1n1, mean_roc_dict_h1n1, fit_params = {'silent': True})


# In[29]:


cross_validate_model_h1n1('STACK', models_h1n1, mean_roc_dict_h1n1)


# In[30]:


mean_roc_df_h1n1 = pd.DataFrame(mean_roc_dict_h1n1, index = ['Model', 'Mean ROC AUC Score']).transpose().sort_values(by='Mean ROC AUC Score', ascending=False)
mean_roc_df_h1n1


# ## SEASONAL Label

# In[31]:


mean_roc_dict_seas = {}


# In[32]:


feature_cols = [col for col in X_train_full_seas.columns.tolist() if col not in ['h1n1_vaccine', 'seasonal_vaccine']]
target_cols = ['seasonal_vaccine']


# In[33]:


cross_validate_model_seas('Logist', models_seas, mean_roc_dict_seas)


# In[34]:


cross_validate_model_seas('KNN', models_seas, mean_roc_dict_seas)


# In[35]:


cross_validate_model_seas('MLP', models_seas, mean_roc_dict_seas)


# In[36]:


cross_validate_model_seas('GaussNB', models_seas, mean_roc_dict_seas)


# In[37]:


cross_validate_model_seas('XGB', models_seas, mean_roc_dict_seas)


# In[38]:


cross_validate_model_seas('LGBM', models_seas, mean_roc_dict_seas)


# In[39]:


cross_validate_model_seas('CAT', models_seas, mean_roc_dict_seas, fit_params = {'silent': True})


# In[ ]:


cross_validate_model_seas('STACK', models_seas, mean_roc_dict_seas)


# In[ ]:


mean_roc_df_seas = pd.DataFrame(mean_roc_dict_seas, index = ['Model', 'Mean ROC AUC Score']).transpose().sort_values(by='Mean ROC AUC Score', ascending=False)
mean_roc_df_seas


# # Plotting Best Model

# In[ ]:


from sklearn.model_selection import train_test_split

X_train_seas, X_valid_seas, y_train_seas, y_valid_seas = train_test_split(X_train_full_seas,
                                                      y_train_full,
                                                      train_size=0.8,
                                                      random_state=42,
                                                      shuffle=True)

X_train_h1n1, X_valid_h1n1, y_train_h1n1, y_valid_h1n1 = train_test_split(X_train_full_h1n1,
                                                      y_train_full,
                                                      train_size=0.8,
                                                      random_state=42,
                                                      shuffle=True)


# In[ ]:


def plot_roc(y_true, y_score, label_name, title_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr, label = f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}")
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(title_name)
    ax.legend(loc='lower right')


# In[ ]:


best_model_h1n1 = mean_roc_df_h1n1.iloc[0]['Model']
best_model_h1n1_name = mean_roc_df_h1n1.index[0]


# In[ ]:


best_model_h1n1.fit(X_train_h1n1, y_train_h1n1['h1n1_vaccine'])


# In[ ]:


y_pred_h1n1 = best_model_h1n1.predict_proba(X_valid_h1n1)[:, 1]


# In[ ]:


best_model_seas = mean_roc_df_seas.iloc[0]['Model']
best_model_seas_name = mean_roc_df_seas.index[0]


# In[ ]:


best_model_seas.fit(X_train_seas, y_train_seas['seasonal_vaccine'])


# In[ ]:


y_pred_seas = best_model_seas.predict_proba(X_valid_seas)[:, 1]


# In[ ]:


y_pred = pd.DataFrame({'h1n1_vaccine': y_pred_h1n1,
                       'seasonal_vaccine': y_pred_seas})


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))

plot_roc(
    y_valid['h1n1_vaccine'], 
    y_pred['h1n1_vaccine'], 
    label_name = best_model_h1n1_name,
    title_name = 'h1n1_vaccine',
    ax=ax[0]
)

plot_roc(
    y_valid['seasonal_vaccine'], 
    y_pred['seasonal_vaccine'], 
    label_name = best_model_seas_name,
    title_name = 'seasonal_vaccine',
    ax=ax[1]
)

mean_auc = (roc_auc_score(y_valid['h1n1_vaccine'], y_pred['h1n1_vaccine']) + roc_auc_score(y_valid['seasonal_vaccine'], y_pred['seasonal_vaccine']))/2
plt.suptitle(f'Mean AUC Score: {mean_auc:.4f}')

fig.tight_layout()


# In[ ]:


pickle.dump(best_model_h1n1, open('models\\estimator_h1n1.pkl', 'wb'))
pickle.dump(best_model_seas, open('models\\estimator_seas.pkl', 'wb'))

