#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle


# In[2]:


df=pd.read_excel('shrimp.xlsx')
df


# In[3]:


x=df[['DOC', 'Salinity (ppt)', 'TSS (ml/L)', 'Average PH', 'Average DO (ppm)', 'TDF (kg)']]
x


# In[4]:


y=df['ABW (gm)']
y



from sklearn.ensemble import RandomForestRegressor


# In[7]:


gbrreg=RandomForestRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=2, random_state=42)
gbrreg.fit(x,y)


# In[8]:


pickle.dump(gbrreg, open('model.pkl','wb'))


# In[9]:


model = pickle.load(open('model.pkl','rb'))


# In[ ]: