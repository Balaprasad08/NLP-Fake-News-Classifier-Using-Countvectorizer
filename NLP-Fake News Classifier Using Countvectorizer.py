#!/usr/bin/env python
# coding: utf-8

# ### Import Important Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
import re
import nltk


# ### Load Dataset

# In[2]:


os.chdir('E:\\prasad\\practice\\NLP\\dataset\\fake-news')


# In[3]:


df=pd.read_csv('train.csv')


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# ### Visualize NA Values

# In[6]:


sns.heatmap(df.isnull())
plt.show()


# In[7]:


df.head(10)


# ### Drop NA Values

# In[8]:


df.dropna(inplace=True)


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


sns.heatmap(df.isnull())
plt.show()


# In[12]:


df.head(10)


# ### Reset Index

# In[13]:


df.reset_index(inplace=True)


# In[14]:


df.head(10)


# In[15]:


messages=df.copy()


# In[16]:


messages.shape


# In[17]:


messages['title'][9]


# In[18]:


messages['title'][5]


# ### Cleaning the texts

# In[19]:


from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
ps=PorterStemmer()


# In[20]:


corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['title'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)


# In[21]:


messages['title'][1]


# In[22]:


corpus[1]


# In[23]:


messages['title'][5]


# In[24]:


corpus[5]


# ### Creating the Bag of Words

# In[25]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer


# In[26]:


# Applying Countvectorizer
cv=CountVectorizer(max_features=5000,ngram_range=(1,3))


# In[27]:


X=cv.fit_transform(corpus).toarray()
X.shape


# In[28]:


y=messages['label']
y.shape


# ### Divide the dataset into Train and Test

# In[29]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[30]:


cv.get_feature_names()[0:10]


# In[31]:


cv.get_params()


# ### Create DataFrame using New Feature Names

# In[32]:


count_df=pd.DataFrame(X,columns=cv.get_feature_names())
count_df.head()


# In[33]:


X.shape


# #### Create Function for Model Building

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[35]:


def check_model(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('Accuracy Score:',accuracy_score(y_test,y_pred),'\n')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test,y_pred),'\n')
    print('Classification Report:')
    print(classification_report(y_test,y_pred))


# In[36]:


check_model(MultinomialNB(),X_train,X_test,y_train,y_test)


# In[37]:


# LogisticRegression
# check_model(LogisticRegression(),X_train,X_test,y_train,y_test)


# In[38]:


# RandomForestClassifier
# check_model(RandomForestClassifier(),X_train,X_test,y_train,y_test)


# In[39]:


# DecisionTreeClassifier
# check_model(DecisionTreeClassifier(),X_train,X_test,y_train,y_test)


# In[40]:


# SVC
# check_model(SVC(),X_train,X_test,y_train,y_test)


# In[41]:


# KNeighborsClassifier
# check_model(KNeighborsClassifier(),X_train,X_test,y_train,y_test)


# In[42]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[43]:


check_model(PassiveAggressiveClassifier(),X_train,X_test,y_train,y_test)


# In[ ]:




