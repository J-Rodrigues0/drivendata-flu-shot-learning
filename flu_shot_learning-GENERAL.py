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
plt.rcParams['figure.dpi'] = 120


# In[2]:


X_train_full = pd.read_csv('input_data\\training_set_features.csv', index_col = 'respondent_id')
y_train_full = pd.read_csv('input_data\\training_set_labels.csv', index_col = 'respondent_id')


# In[3]:


X_test = pd.read_csv('input_data\\test_set_features.csv', index_col = 'respondent_id')


# # Loading Help Functions

# In[4]:


def simplify_col_names(df):
    df['income_poverty'].replace('Below Poverty', 'Low', inplace = True)
    df['income_poverty'].replace('<= $75,000, Above Poverty', 'Medium', inplace = True)
    df['income_poverty'].replace('> $75,000', 'High', inplace = True)
    
    df['age_group'].replace('65+ Years', '65+', inplace = True)
    df['age_group'].replace('55 - 64 Years', '55+', inplace = True)
    df['age_group'].replace('45 - 54 Years', '45+', inplace = True)
    df['age_group'].replace('35 - 44 Years', '35+', inplace = True)
    df['age_group'].replace('18 - 34 Years', '18+', inplace = True)
    
    df['education'].replace('College Graduate', 'Very High', inplace = True)
    df['education'].replace('Some College', 'High', inplace = True)
    df['education'].replace('12 Years', 'Medium', inplace = True)
    df['education'].replace('< 12 Years', 'Low', inplace = True)


# In[5]:


def engineer_features(df):
    behavioral_cols = ['behavioral_antiviral_meds',
                   'behavioral_avoidance',
                   'behavioral_face_mask',
                   'behavioral_wash_hands',
                   'behavioral_large_gatherings',
                   'behavioral_outside_home',
                   'behavioral_touch_face']

    #df['general_behavior'] = pd.Series(np.zeros(df.shape[0]), index = df.index)
    #for b_col in behavioral_cols:
    #    df['general_behavior'] += df[b_col]
    
    df['general_effective'] = df['opinion_h1n1_vacc_effective'] + df['opinion_seas_vacc_effective']

    df['general_risk'] = df['opinion_h1n1_risk'] + df['opinion_seas_risk']
    
    df['general_eff_risk'] = df['general_effective'] + df['general_risk']
    
    df['general_reccomendation'] = df['doctor_recc_h1n1'] + df['doctor_recc_seasonal']
    
    df['general_h1n1_info'] = df['h1n1_concern'] + df['h1n1_knowledge']
    
    df['household_total'] = df['household_adults'] + df['household_children'] + df['child_under_6_months'] + 1


# # Loading Models

# In[6]:


preprocessor_seas = pickle.load(open('models\\preprocessor_seas.pkl', 'rb'))
preprocessor_h1n1 = pickle.load(open('models\\preprocessor_h1n1.pkl', 'rb'))


# In[7]:


estimator_h1n1 = pickle.load(open('models\\estimator_h1n1.pkl', 'rb'))
estimator_seas = pickle.load(open('models\\estimator_seas.pkl', 'rb'))


# # Modelling

# Build final pipelines:

# In[8]:


from sklearn.pipeline import Pipeline

full_pipeline_h1n1 = Pipeline(steps=[('preprocessor', preprocessor_h1n1),
                                     ('estimator', estimator_h1n1)])

full_pipeline_seas = Pipeline(steps=[('preprocessor', preprocessor_seas),
                                     ('estimator', estimator_seas)])


# Prepare input data:

# In[9]:


simplify_col_names(X_train_full)


# In[10]:


engineer_features(X_train_full)


# Fit pipeline to full train data:

# In[11]:


full_pipeline_h1n1.fit(X_train_full, y_train_full['h1n1_vaccine'])


# In[12]:


full_pipeline_seas.fit(X_train_full, y_train_full['seasonal_vaccine'])


# # Making predictions

# In[13]:


simplify_col_names(X_test)


# In[14]:


engineer_features(X_test)


# In[15]:


pred_h1n1 = full_pipeline_h1n1.predict_proba(X_test)[:, 1]
pred_seas = full_pipeline_seas.predict_proba(X_test)[:, 1]

predictions = pd.DataFrame({'respondent_id': X_test.index,
                            'h1n1_vaccine': pred_h1n1,
                            'seasonal_vaccine': pred_seas
                           })


# In[16]:


predictions


# In[17]:


predictions.to_csv('output_data\\predictions.csv', index = False)

