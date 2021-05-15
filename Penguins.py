#!/usr/bin/env python
# coding: utf-8

# # XGBoost Notebook

# ## Quick notebook to practice using xgboost algorithm
# 
# ### The first section of this notebook will do some data analysis on a dataset for penguins (my favourite animal species, after all). After that I will attempt to predict certain features of the dataset using xgboost algorithm.

# Load libraries

# In[200]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error


# Load & inspect dataframe

# In[105]:


df = pd.read_csv('Datasets/penguins.csv')


# In[106]:


df.head(10)


# In[107]:


df.describe()


# In[108]:


df.info()


# In[109]:


df.isna().sum()


# Drop Comments column and fill na values

# In[110]:


df.drop('Comments', axis=1, inplace=True)


# In[111]:


df.fillna(method='bfill', inplace=True)


# In[112]:


df.isna().sum()


# In[113]:


df.nunique()


# In[114]:


df.drop(['Region', 'Stage'], axis=1, inplace=True) #delete as only 1


# Visualisations based on species

# In[115]:


sns.countplot(data=df, x='Species')


# In[116]:


sns.boxplot(data=df, x='Species', y='Culmen Length (mm)')


# In[117]:


sns.violinplot(data=df, x='Species', y='Culmen Length (mm)')


# In[118]:


sns.boxplot(data=df, x='Species', y='Culmen Depth (mm)')


# In[119]:


sns.boxplot(data=df, x='Species', y='Body Mass (g)')


# Visualisations based on sex

# In[120]:


sns.countplot(data=df, x='Sex')


# In[121]:


sns.boxplot(data=df, x='Sex', y='Culmen Length (mm)')


# In[122]:


sns.violinplot(data=df, x='Species', y='Culmen Length (mm)')


# In[123]:


sns.boxplot(data=df, x='Sex', y='Culmen Depth (mm)')


# In[124]:


sns.boxplot(data=df, x='Species', y='Body Mass (g)')


# Plot the emperical distribution function

# In[125]:


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / n
    return x, y


# In[126]:


df.Species.unique()


# In[127]:


x_AP, y_AP = ecdf(df[df.Species == 'Adelie Penguin (Pygoscelis adeliae)']['Culmen Length (mm)'])
x_GP, y_GP = ecdf(df[df.Species == 'Gentoo penguin (Pygoscelis papua)']['Culmen Length (mm)'])
x_CP, y_CP = ecdf(df[df.Species == 'Chinstrap penguin (Pygoscelis antarctica)']['Culmen Length (mm)'])


# In[128]:


_ = plt.plot(x_AP, y_AP, marker='.', linestyle='none')
_ = plt.plot(x_GP, y_GP, marker='.', linestyle='none')
_ = plt.plot(x_CP, y_CP, marker='.', linestyle='none')
# Label the axes
_ = plt.legend(('Adelie Penguin', 'Gentoo penguin', 'Chinstrap penguin'))
_ = plt.ylabel('ECDF')
_ = plt.xlabel('Culmen Length (mm)')
_ = plt.title('Emperical Distribution Function')
plt.show()


# # XGBoost

# ## Start using xgboost classifier to predict sex of penguin

# Drop unnecessary columns

# In[129]:


df.columns


# In[130]:


df.drop(['Unnamed: 0', 'studyName', 'Sample Number', 'Individual ID', 'Date Egg'], axis=1, inplace=True)


# In[131]:


df.head()


# In[147]:


features_df = df.drop('Sex', axis=1)


# In[148]:


label_col = ['Sex']
label = df[label_col]


# In[149]:


categorical_mask = (features_df.dtypes == object)


# In[150]:


categorical_columns = features_df.columns[categorical_mask].tolist()


# In[151]:


print(df[categorical_columns].head())


# In[152]:


le = LabelEncoder()


# In[153]:


features_df = pd.get_dummies(features_df)


# In[154]:


label = label.apply(lambda x: le.fit_transform(x))


# In[156]:


features_df


# In[157]:


X = features_df.values
y = label.values


# In[158]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[159]:


xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)


# In[ ]:


xg_cl.fit(X_train, y_train)


# In[162]:


preds = xg_cl.predict(X_test)


# In[163]:


accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


# In[165]:


churn_dmatrix = xgb.DMatrix(data=X_train, label=y_train)


# In[166]:


params = {"objective":"reg:logistic", "max_depth":3}


# In[167]:


cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                  nfold=3, num_boost_round=5, 
                  metrics="error", as_pandas=True, seed=123)


# In[172]:


print(cv_results)
print('--------')
print('Accuracy: '+str((1-cv_results["test-error-mean"]).iloc[-1]))


# ### Now onto a regression model, lets try and predict the culmen length

# In[173]:


reg_features_df = df.drop('Culmen Length (mm)', axis=1)


# In[174]:


reg_label_col = ['Culmen Length (mm)']
reg_label = df[reg_label_col]


# In[176]:


reg_categorical_mask = (reg_features_df.dtypes == object)
reg_categorical_columns = reg_features_df.columns[reg_categorical_mask].tolist()


# In[177]:


reg_features_df = pd.get_dummies(reg_features_df)


# In[178]:


reg_label = reg_label.apply(lambda x: le.fit_transform(x))


# In[179]:


X = reg_features_df.values
y = reg_label.values


# In[180]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[181]:


xg_reg = xgb.XGBRegressor(seed=123, objective='reg:linear', n_estimators=10)


# In[ ]:


xg_reg.fit(X_train, y_train)


# In[184]:


reg_preds = xg_reg.predict(X_test)


# In[188]:


rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# In[191]:


DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test =  xgb.DMatrix(data=X_test, label=y_test)


# In[192]:


params = {"booster":"gblinear", "objective":"reg:linear"}


# In[193]:


xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)


# In[194]:


preds = xg_reg.predict(DM_test)


# In[195]:


rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))


# Add some fine tuning

# In[197]:


gbm_param_grid = {
    'n_estimators': [5, 10, 15, 20, 25],
    'max_depth': range(2, 20)
}


# In[198]:


gbm = xgb.XGBRegressor(n_estimators=10)


# In[201]:


randomized_mse = RandomizedSearchCV(gbm, 
                                    param_distributions=gbm_param_grid, 
                                    scoring='neg_mean_squared_error', 
                                    n_iter=5, cv=4, verbose=1)


# In[202]:


randomized_mse.fit(X_train, y_train)


# In[203]:


print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))


# In[ ]:




