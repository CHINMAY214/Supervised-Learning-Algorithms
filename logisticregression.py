#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[19]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression


# # Reading The Dataset

# In[20]:


diabetes=pd.read_csv("diabetes.csv")


# In[21]:


diabetes


# In[22]:


#First five entries of dataset
diabetes.head()


# In[23]:


#Last five entries of dataset
diabetes.tail()


# # Data Cleaning

# In[24]:


#To check the null values
diabetes.isnull().sum()


# In[25]:


# Calculating the median of insulin column
dia_med = diabetes['Insulin'].median()
# Replacing the null values with median values
diabetes['Insulin'].replace(0,dia_med,inplace=True)


# In[26]:


#Checkin weather any value is null or not
diabetes[diabetes.Insulin==0]


# In[27]:


# Calculating the median of SkinThickness column 
d1 = diabetes['SkinThickness'].median()
# Replacing the null values with median values
diabetes['SkinThickness'].replace(0,d1,inplace=True)


# In[28]:


diabetes.head()


# In[29]:


# Plotting KDE plot
for i in diabetes:
    sns.kdeplot(data=diabetes,x=i)
    plt.grid()
    plt.show()


# In[30]:


sns.pairplot(diabetes)
plt.show()


# In[12]:


#plottig heatmap for checking corelation
sns.heatmap(diabetes.corr(),annot=True)


# # Splitting the data

# In[31]:


x=diabetes.drop('Outcome',axis=1)
y=diabetes['Outcome']


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)


# # Scaling the Data

# In[33]:


SS=StandardScaler()


# In[34]:


x_train=SS.fit_transform(x_train)


# In[35]:


x_test=SS.transform(x_test)


# # Model Training

# In[36]:


LR=LogisticRegression()


# In[37]:


LR.fit(x_train,y_train)


# In[42]:


pred = LR.predict(x_test)


# In[39]:


y_test


# # Model Evaluation

# In[40]:


LR.score(x_train,y_train)  # positive class


# In[41]:


LR.score(x_test,y_test)    # positive class


# In[55]:


AC = accuracy_score(y_test,pred)*100
print("The accuracy Score in percentage:")
print(AC)


# In[54]:


confusion_matrix(y_test,pred)


# In[53]:


print(classification_report(pred,y_test))


# In[ ]:




