#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


#importing dateset 
df=pd.read_csv(r'medical_records_ml.csv')
df 


# In[3]:


#chgecking any null values are present or not in our dataset 
df.info ()
# so there is no any null values 


# In[4]:


################################################ here we start eda ###############################################
# 1.Genderwise patient count 
df.groupby('gender').count()['patient_id'].reset_index()


# In[5]:


#here in our dataset there is one insights i got that male patients are more than female patient.


# In[6]:


df["Year"]=df["Year"].astype(object)


# In[7]:


#here we change our year datatype we change into int to object 
df.info()


# In[8]:


#yearwise patientcount
df.groupby("Year").count()["gender"].reset_index()


# In[9]:


#As we seen above we have maximum patient is in year 2022


# In[13]:


#monthwise patient count
df.groupby("month_number").count()["patient_id"].reset_index()


# In[14]:


#As we can see in monthwise patient not big difference found in this eda there are all patient count are same in all month there  atre some slightly differences.


# In[19]:


cols=["male&female_0_1","patient_id"]
new_df=df[cols]
new_df.head()


# In[32]:


#creating ML model 
#Here we create k mean clustering algorithm 
from sklearn.cluster import KMeans 
kmeans=KMeans(n_clusters=4)


# In[33]:


kmeans.fit(new_df)


# In[34]:


clusters=kmeans.predict(new_df)


# In[35]:


new_df["clusters"]=clusters
new_df


# In[36]:


new_df[new_df["clusters"]==1]


# In[ ]:


#As we see our model kmeans algorithm divide the data into 4 clusters depends similarities it is easy we took count of similarities.
#and also easy to see patient count of specific clusters and give the next appointment date. to our patients.


# In[ ]:





# In[ ]:




