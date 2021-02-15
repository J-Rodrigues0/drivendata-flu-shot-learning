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


# # Loading data

# In[2]:


X_train_full = pd.read_csv('input_data\\training_set_features.csv', index_col = 'respondent_id')
y_train_full = pd.read_csv('input_data\\training_set_labels.csv', index_col = 'respondent_id')


# In[3]:


X_test = pd.read_csv('input_data\\test_set_features.csv', index_col = 'respondent_id')


# # Data First Impressions

# In[4]:


X_train_full.shape


# In[5]:


X_train_full.head()


# In[6]:


y_train_full.head()


# In[7]:


X_train_full.dtypes


# In[8]:


float_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == 'float64']


# In[9]:


X_train_full[float_cols[:len(float_cols)//2]].describe()


# In[10]:


X_train_full[float_cols[len(float_cols)//2:]].describe()


# * We can see that most columns are binary (either 0 or 1) - this can be seen from the quartiles of the data.
# * We can also immediately see that some columns have missing values - count < data.shape[0].

# # Data Cleaning

# Lets see which columns have missing values:

# In[11]:


missing_values_df = pd.DataFrame({'Missing Absolute': X_train_full.isna().sum(),
              'Missing Percentage': np.round(X_train_full.isna().sum()/X_train_full.shape[0] * 100, 2)})
missing_values_df


# Lets look more closely to the features with more than 10% missing values:

# In[12]:


missing_values_df[missing_values_df['Missing Percentage'] > 10]


# These columns have such a high percentage of missing values that I might consider just dropping them out of the classification.
# However, I believe these columns could have a big impact on the success of the algorithm because:
# 
# * *health_insurance* - health insurance could cover the vacine costs and also the treatment so it influences the person's decision to take the shot;
# * *income_poverty* - lower income people probably tend to not get the vaccine as often as higher income people;
# * *employment_industry/occupation* - a person working in the health sector is probably more likely to get the shots than someone from other industries.
# 
# Thus, I believe we should make an effort to fill the missing values in these columns.

# For the *income_poverty* feature, the possible values are:

# In[13]:


X_train_full['income_poverty'].value_counts()


# Which we can simplify in terms of income as [*Low*, *Medium*, *High*]:

# In[14]:


# X_train_full['income_poverty'].replace('Below Poverty', 'Low', inplace = True)
# X_train_full['income_poverty'].replace('<= $75,000, Above Poverty', 'Medium', inplace = True)
# X_train_full['income_poverty'].replace('> $75,000', 'High', inplace = True)


# We should design a function to make this simpification more general so that we can later apply it to the test dataset:

# In[15]:


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


# In[16]:


simplify_col_names(X_train_full)


# Now, from our EDA, we can see that income has a relation with taking the vaccines so we should try to *OrdinalEncode* this column:

# In[17]:


from sklearn.impute import SimpleImputer

imputer_most_frequent = SimpleImputer(strategy='most_frequent')


# In[18]:


from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

income_pipe = Pipeline(steps=[('impute', imputer_most_frequent),
                              ('encode', OrdinalEncoder(categories=[np.array(['Low', 'Medium', 'High'])]))
                             ])

missing_values_df.drop('income_poverty', axis = 0, inplace = True)

ordinal_cols_seas = ['income_poverty']
ordinal_cols_h1n1 = ['income_poverty']


# We will now do the same for the *education* and *age_group* columns - also based on our EDA:

# In[19]:


education_pipe = Pipeline(steps=[('impute', imputer_most_frequent),
                                 ('encode', OrdinalEncoder(categories=[np.array(['Low', 'Medium', 'High', 'Very High'])]))
                                ])

age_pipe = Pipeline(steps=[('impute', imputer_most_frequent),
                           ('encode', OrdinalEncoder(categories=[np.array(['18+', '35+', '45+', '55+', '65+'])]))
                          ])

ordinal_cols_seas.extend(['education', 'age_group'])
ordinal_cols_h1n1.extend(['education', 'age_group'])


# Regarding, *health_insurance* and *employment_industry/occupation*, my first idea was that, taking into account that almost half of the data is missing, I was going to drop these columns, instead of making possibly biased assumptions about the data. However, I later experimented with adding these columns to the *categorical_cols* preprocessing and got better results so that's what I will do.

# In[20]:


cols_drop = ['health_insurance', 'employment_industry', 'employment_occupation']
missing_values_df.drop(cols_drop, axis = 0, inplace = True)


# In[21]:


industry_order_seas = ['rucpziij',
                     'xicduogh',
                     'saaquncn',
                     'mcubkhph',
                     'nduyfdeo',
                     'vjjrobsf',
                     'xqicxuve',
                     'phxvnwax',
                     'pxcmvdjn',
                     'dotnnunm',
                     'cfqqtusy',
                     'atmlpfrs',
                     'ldnlellj',
                     'wlfvacwt',
                     'mfikgejo',
                     'wxleyezf',
                     'msuufmds',
                     'arjwrbjb',
                     'qnlwzans',
                     'fcxhlnwr',
                     'haxffmxo']

occupation_order_seas = ['qxajmpny',
                         'uqqtjvyb',
                         'tfqavkke',
                         'pvmttkik',
                         'rcertsgn',
                         'xqwwgdyp',
                         'xgwztkwe',
                         'ukymxvdu',
                         'ccgxvspp',
                         'vlluhbov',
                         'oijqvulv',
                         'kldqjyjy',
                         'xtkaffoo',
                         'dlvbwzss',
                         'hfxkjkmi',
                         'mxkfnird',
                         'hodpvpew',
                         'bxpfxfdn',
                         'emcorrxb',
                         'xzmlyyjv',
                         'haliazsg',
                         'cmhcxjea',
                         'dcjcmpih']


# In[22]:


industry_order_h1n1 = ['dotnnunm',
                     'xicduogh',
                     'atmlpfrs',
                     'mcubkhph',
                     'vjjrobsf',
                     'wlfvacwt',
                     'pxcmvdjn',
                     'xqicxuve',
                     'rucpziij',
                     'cfqqtusy',
                     'msuufmds',
                     'mfikgejo',
                     'phxvnwax',
                     'nduyfdeo',
                     'saaquncn',
                     'ldnlellj',
                     'qnlwzans',
                     'arjwrbjb',
                     'wxleyezf',
                     'fcxhlnwr',
                     'haxffmxo']

occupation_order_h1n1 = ['rcertsgn',
                         'qxajmpny',
                         'uqqtjvyb',
                         'xgwztkwe',
                         'xqwwgdyp',
                         'pvmttkik',
                         'tfqavkke',
                         'ccgxvspp',
                         'hfxkjkmi',
                         'mxkfnird',
                         'oijqvulv',
                         'ukymxvdu',
                         'vlluhbov',
                         'xzmlyyjv',
                         'xtkaffoo',
                         'kldqjyjy',
                         'emcorrxb',
                         'hodpvpew',
                         'bxpfxfdn',
                         'dlvbwzss',
                         'haliazsg',
                         'cmhcxjea',
                         'dcjcmpih']


# In[23]:


industry_pipe_seas = Pipeline(steps=[('impute', imputer_most_frequent),
                                 ('encode', OrdinalEncoder(categories=[np.array(industry_order_seas)]))
                                ])

occupation_pipe_seas = Pipeline(steps=[('impute', imputer_most_frequent),
                           ('encode', OrdinalEncoder(categories=[np.array(occupation_order_seas)]))
                          ])

ordinal_cols_seas.extend(['employment_industry', 'employment_occupation'])


# In[24]:


industry_pipe_h1n1 = Pipeline(steps=[('impute', imputer_most_frequent),
                                 ('encode', OrdinalEncoder(categories=[np.array(industry_order_h1n1)]))
                                ])

occupation_pipe_h1n1 = Pipeline(steps=[('impute', imputer_most_frequent),
                           ('encode', OrdinalEncoder(categories=[np.array(occupation_order_h1n1)]))
                          ])

#ordinal_cols_h1n1.extend(['employment_industry', 'employment_occupation'])


# Now, we can look into two other columns with missing values (> 8% missing) - *doctor_recc_h1n1* and *doctor_recc_seasonal*:

# In[25]:


missing_values_df[missing_values_df['Missing Percentage'] > 8]


# In[26]:


pd.DataFrame({'Recc H1N1': X_train_full['doctor_recc_h1n1'].value_counts(),
              'Recc SEAS': X_train_full['doctor_recc_seasonal'].value_counts()})


# Since doctors seem to not recommend the vacines about 2-3 times as often as they recommend it, we can assume that the missing values represent times where the doctor did not recommend the shots. So:

# In[27]:


doctor_recc_list = ['doctor_recc_h1n1', 'doctor_recc_seasonal']

cols_most_frequent = []
cols_most_frequent.extend(doctor_recc_list)

missing_values_df.drop(doctor_recc_list, axis = 0, inplace = True)


# For the remaining columns, we can also impute them with the *most frequent* value since there are <8% of missing values and the values are mostly categorical so we dont want to impute with the mean. Thus: 

# In[28]:


cols_most_frequent.extend(list(missing_values_df[missing_values_df > 0].index))


# In[29]:


categorical_cols_seas = [col for col in X_train_full.columns if X_train_full[col].dtype == 'object' 
                         and col not in cols_drop and col not in ordinal_cols_seas]

categorical_cols_h1n1 = [col for col in X_train_full.columns if X_train_full[col].dtype == 'object' 
                         and col not in cols_drop and col not in ordinal_cols_h1n1]


# In[30]:


categorical_cols_seas.extend(cols_drop)
categorical_cols_h1n1.extend(cols_drop)


# In[31]:


from sklearn.preprocessing import OneHotEncoder

cat_preprocessor = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                             ('encode', OneHotEncoder(handle_unknown='ignore', sparse=False))
                            ])


# In[32]:


numerical_cols_seas = [col for col in X_train_full.columns if col not in categorical_cols_seas
                       and col not in cols_drop and col not in ordinal_cols_seas]

numerical_cols_h1n1 = [col for col in X_train_full.columns if col not in categorical_cols_h1n1
                       and col not in cols_drop and col not in ordinal_cols_h1n1]


# # Feature Engineering

# Using the conclusions of our EDA, we want to engineer the following features:
# * *general_behavior*
# * *general_effective*
# * *general_risk*
# * *general_eff_risk*
# * *general_reccomendation*
# 
# *general_behaviour* was later observed to worsen the results of our model so we are not engineering that feature. The code is left commented, for reference.

# In[33]:


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


# In[34]:


engineer_features(X_train_full)


# In[35]:


engineered_features = ['general_effective',
                       'general_risk',
                       'general_eff_risk',
                       'general_reccomendation',
                       'general_h1n1_info',
                       'household_total']

numerical_cols_seas.extend(engineered_features)
numerical_cols_h1n1.extend(engineered_features)


# # Final preprocessor
# 
# Preprocessing can be made convenient using *scikit learn* and a *ColumnTransformer*:

# In[36]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

ColTransf_seas = ColumnTransformer([('numerical_cols', imputer_most_frequent, numerical_cols_seas),
                                  ('categorical_cols', cat_preprocessor, categorical_cols_seas),
                                  ('income', income_pipe, ['income_poverty']),
                                  ('education', education_pipe, ['education']),
                                  ('age', age_pipe, ['age_group']),
                                  ('industry', industry_pipe_seas, ['employment_industry']),
                                  ('occupation', occupation_pipe_seas, ['employment_occupation'])])

ColTransf_h1n1 = ColumnTransformer([('numerical_cols', imputer_most_frequent, numerical_cols_h1n1),
                                  ('categorical_cols', cat_preprocessor, categorical_cols_h1n1),
                                  ('income', income_pipe, ['income_poverty']),
                                  ('education', education_pipe, ['education']),
                                  ('age', age_pipe, ['age_group'])])

preprocessor_seas = Pipeline(steps=[('ColTransf', ColTransf_seas),
                               ('Scaler', StandardScaler())])

preprocessor_h1n1 = Pipeline(steps=[('ColTransf', ColTransf_h1n1),
                               ('Scaler', StandardScaler())])


# In[37]:


X_train_preprocessed = pd.DataFrame(preprocessor_seas.fit_transform(X_train_full), index = X_train_full.index)
X_train_preprocessed.to_csv('interim_data\\preprocessed_train_features_seas.csv')


# In[38]:


X_train_preprocessed = pd.DataFrame(preprocessor_h1n1.fit_transform(X_train_full), index = X_train_full.index)
X_train_preprocessed.to_csv('interim_data\\preprocessed_train_features_h1n1.csv')


# In[39]:


pickle.dump(preprocessor_seas, open('models\\preprocessor_seas.pkl', 'wb'))
pickle.dump(preprocessor_h1n1, open('models\\preprocessor_h1n1.pkl', 'wb'))

