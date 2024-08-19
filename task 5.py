#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Advertising.csv')
df.head()


# In[3]:


# Drop the index column if necessary becouse its not much need
df = df.drop(columns=['Unnamed: 0'])


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[8]:


df.isnull()


# In[9]:


df.isnull().sum()


# In[10]:


df.boxplot()


# In[11]:


df.corr()


# In[12]:


df.head()


# In[13]:


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


# In[ ]:





# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[17]:


X_train


# In[19]:


from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)


# In[20]:


from sklearn.metrics import mean_squared_error, r2_score


# In[21]:


mse = mean_squared_error(y_test, y_pred)
mse


# In[22]:


r2 = r2_score(y_test, y_pred)
r2


# In[24]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actuals vs Predicted Sales')
plt.show()


# In[25]:


# Print the coefficients
coefficients = model.coef_
features = X.columns
for feature, coef in zip(features, coefficients):
    print(f'{feature}: {coef}')


# In[ ]:




