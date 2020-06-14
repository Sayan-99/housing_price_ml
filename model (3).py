#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# In[27]:


df = pd.read_csv('house_predict.csv')


# In[28]:


df.head()


# In[29]:


df.columns


# In[30]:


feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']


# In[31]:


X = df[feature_names]


# In[32]:


X.head()


# In[33]:


y = df['SalePrice']


# In[34]:


y


# In[35]:


X.isnull().sum()


# In[36]:


X.astype('int64')


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.75,random_state = 0)


# In[44]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 500,random_state = 1)
regressor.fit(X_train,y_train)


# In[45]:


y_pred = regressor.predict(X_test)


# In[46]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)


# In[47]:


pickle_out = open('regressor.pkl','wb')
pickle.dump(regressor,pickle_out)
pickle_out.close()


# In[48]:


regressor.predict([[8450,2003,854,562,2,5,8]])


# In[ ]:





# In[ ]:





# In[ ]:




