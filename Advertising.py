#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd


# In[13]:


import seaborn as sns


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


ad_data = pd.read_csv('advertising.csv')


# In[17]:


ad_data.head()


# In[20]:


ad_data.info()


# In[21]:


ad_data.describe()


# In[22]:


ad_data['Age'].plot.hist(bins=35)


# In[23]:


sns.jointplot(x='Age',y='Area Income', data=ad_data)


# In[24]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red')


# In[ ]:


sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')


# In[19]:




sns.pairplot(ad_data,hue='Clicked on Ad')


# In[35]:



from sklearn.model_selection import train_test_split


# In[25]:


ad_data.head()


# In[36]:


X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']


# In[37]:


X_train, X_test, y_train,   y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[38]:


# Traunning is done of logistic regression (fit a logistic regression model on the trainning set)


# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


logmodel = LogisticRegression()


# In[41]:


logmodel.fit(X_train,y_train)


# In[42]:


# Predictions and Evaluation is done ()


# In[49]:


predictions = logmodel.predict(X_test)


# In[50]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:




