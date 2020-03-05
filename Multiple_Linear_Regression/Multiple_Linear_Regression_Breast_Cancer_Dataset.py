#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

read_csv = pd.read_csv("data.csv")


# In[2]:


read_csv.isnull().sum()


# In[3]:


read_csv.dropna()


# In[4]:


X = read_csv.iloc[ :,2:-1].values
Y = read_csv.iloc[ :,1:2].values


# In[5]:


X


# In[6]:


# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
Y = label_encoder.fit_transform(Y)

X


# In[7]:


Y


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size =0.2)


# In[10]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# Predicting the Test set results
y_pred = regressor.predict(X_Test)


# In[43]:


Predicted = [int(round(i)) for i in y_pred]

Predicted


# In[44]:


from sklearn.metrics import accuracy_score


accuracy = accuracy_score(Y_Test,Predicted)
# ac = accuracy_score(Y_Test, regressor.predict(X_Test), normalize=False)
accuracy


# In[38]:


# data = [0.77605316,  0.29409849,  0.82718746, -0.11478048]


# In[39]:


from sklearn.metrics import confusion_matrix


# In[41]:


confusion_matrix(Y_Test, ss)


# In[ ]:





# In[ ]:




