#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[4]:


buyers=pd.read_csv("Indian automobile buying behaviour study 1.0.csv")
location=pd.read_csv("Location_data.csv")


# In[10]:


buyers.head()


# In[11]:


location.head()


# In[12]:


buyers.describe()


# In[13]:


location.describe()


# In[14]:


pd.DataFrame(location.drop(axis=0, columns='State Name').sum())


# In[15]:


plt.figure(figsize=(12,5))
plt.title('State vs Total Electric Vehicles')
ax=sns.barplot(x='State Name',y='Total Electric Vehicle',data=location)
plt.setp(ax.get_xticklabels(),rotation=75)
plt.show()


# In[16]:


plt.figure(figsize=(15,5))
plt.title('State vs Total Electric Vehicles')
ax=sns.barplot(x='State Name',y='Total Non-Electric Vehicle',data=location)
plt.setp(ax.get_xticklabels(),rotation=75)
plt.show()


# In[17]:


location['Percentage 1']=location['Total Electric Vehicle']/location['Total']


# In[18]:


plt.figure(figsize=(15,5))
plt.title('State vs Percentage of Electric Vehicles')
ax1=sns.barplot(x='State Name',y='Percentage 1',data=location)
plt.setp(ax1.get_xticklabels(),rotation=90)
plt.show()


# In[19]:


location['Percentage 2']=location['Total Non-Electric Vehicle']/location['Total']


# In[20]:


plt.figure(figsize=(15,5))
plt.title('State vs Percentage of Electric Vehicles')
ax2=sns.barplot(x='State Name',y='Percentage 2',data=location)
plt.setp(ax2.get_xticklabels(),rotation=90)
plt.show()


# In[21]:


charging_stations = pd.read_csv('charging_stations_state_wise.csv')


# In[22]:


charging_stations.head()


# In[23]:


plt.figure(figsize=(15,5))
plt.title('State vs Total Charging Stations')
ax=sns.barplot(x='State/UT',y='No. of Operational PCS',data=charging_stations)
plt.setp(ax.get_xticklabels(),rotation=90)
plt.grid(axis='y', linestyle='--')
plt.show()


# In[24]:


Sold_Evs_India = pd.read_csv('sold_EVs.csv')


# In[25]:


Sold_Evs_India.head()


# In[26]:


wheeler_types = Sold_Evs_India['Wheeler Type']
total_vehicles = Sold_Evs_India['Total No. of Vehicle']

plt.figure(figsize=(8, 6))
bars = plt.bar(wheeler_types, total_vehicles, color='red')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval), va='bottom', ha='center', color='black', fontsize=9)

plt.title('Total Number of Electric Vehicles by Wheeler Type')
plt.xlabel('Wheeler Type')
plt.ylabel('Total Number of Vehicles')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


# In[27]:


statewise_vehicles = pd.read_csv('state_wise_EVs.csv')


# In[28]:


statewise_vehicles.head()


# In[29]:


states = statewise_vehicles['State/UT']
total_registered = statewise_vehicles['Till date State wise - Total Number of Vehicles Registered']
total_ev_registered = statewise_vehicles['Till date State wise - Total Vehicle Registered as Electric']

plt.figure(figsize=(12,8))
plt.bar(states, total_registered, color='lightblue', label='Total Registered Vehicles')
plt.bar(states, total_ev_registered, color='darkblue', label='Total Electric Vehicles')
plt.yscale('log')
plt.title('Total Registered Vehicles vs Total Electric Vehicles by State/UT')
plt.xlabel('State/UT')
plt.ylabel('Number of Vehicles')
plt.xticks(rotation=90)
plt.legend()
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


# In[146]:


State_sales = pd.read_csv("Vechiles_Data.csv")


# In[147]:


State_sales = State_sales.drop(State_sales.index[33],axis=0)


# In[148]:


State_sales.head(n=40)


# In[149]:


df1 = State_sales["2020"].sum()


# In[150]:


df1


# In[151]:


df2 = State_sales["2021"].sum()


# In[152]:


df2


# In[153]:


df3 = State_sales["2022"].sum()


# In[154]:


df3


# In[155]:


df4 = State_sales["2023"].sum()


# In[156]:


df4


# In[157]:


plt.figure(figsize=(15,5))
plt.title('State vs 2023 Vechiles Number')
ax=sns.barplot(x='State/UT',y='2023',data=State_sales)
plt.setp(ax.get_xticklabels(),rotation=90)
plt.show()


# In[166]:


buyers.head(n=10)


# In[159]:


plt.figure(figsize=(10,5))
sns.countplot(buyers['Make'])


# In[164]:


plt.figure(figsize=(10,5))
sns.boxplot(data=buyers,x='Price',y='Make')


# In[171]:


buyers.describe()


# In[173]:


ax= plt.figure(figsize=(15,8))
sns.heatmap(buyers.corr(),linewidths=1,linecolor='white',annot=True)


# In[174]:


sns.pairplot(buyers)


# In[ ]:




