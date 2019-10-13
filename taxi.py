#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# In[4]:


import csv


# In[5]:


taxi_df = pd.read_csv('nyc_taxi_data_2014.csv')


# In[7]:


taxi_df.shape


# In[8]:


taxi_df.columns


# In[9]:


taxi_df.dtypes


# In[12]:


taxi_df.head(5)


# In[13]:


taxi_df.describe()


# In[ ]:


#There are no numerical columns with missing data
#The passenger count varies between 1 and 208 with most people number of people being 1 or 2
#The trip duration varying from 0 to 100km with most distance of 1 to 3.08 km. 


# In[16]:


non_num_cols=['vendor_id','pickup_datetime','dropoff_datetime','store_and_fwd_flag']
print(taxi_df[non_num_cols].count())


# In[ ]:


#there are some missing values for store_and_fwd_flag


# In[ ]:


taxi_df['pickup_datetime']


# In[20]:


#Convert pick up datetime and dropoff datetime to datetime format.
taxi_df['pickup_datetime']=pd.to_datetime(taxi_df['pickup_datetime'])
taxi_df['dropoff_datetime']=pd.to_datetime(taxi_df['dropoff_datetime'])


# In[30]:


#passenger count
ax=sns.distplot(taxi_df['passenger_count'],kde=False)
ax.set(xlim=(1, 9))

plt.title('Distribution of Passenger Count')

plt.show()


# In[ ]:


#Here we see that the mostly 1 or 4 passengers avail the cab. 
#The instance of large group of people travelling together is rare.


# In[32]:


taxi_df['pickup_datetime'].nunique()


# In[33]:


taxi_df['dropoff_datetime'].nunique()


# In[34]:


#The returned values are 2545052 and 2548337. 
#This shows that there are many different pickup and drop off dates in these 2 columns.


# In[41]:


taxi_df['pickup_day']=taxi_df['pickup_datetime'].dt.day_name()
taxi_df['dropoff_day']=taxi_df['dropoff_datetime'].dt.day_name()


# In[42]:


taxi_df['pickup_day'].value_counts()


# In[43]:


taxi_df['dropoff_day'].value_counts() 


# In[48]:


#The distribution of days of the week can be seen graphically as well
figure,ax=plt.subplots(nrows=2,ncols=1,figsize=(10,10))
sns.countplot(x='pickup_day',data=taxi_df,ax=ax[0])
ax[0].set_title('Number of Pickups done on each day of the week')
sns.countplot(x='dropoff_day',data=taxi_df,ax=ax[1])
ax[1].set_title('Number of dropoffs done on each day of the week')
plt.tight_layout()


# In[ ]:


#The graphs denote the average estimate of a trip for each day of the week.
#Thus the highest avg time taken to complete a trip is on Thursday 
#while Monday, Tuesday, Saturday and Sunday takes the least time.


# In[55]:


#The distribution of Pickup and Drop Off hours of the day

def timezone(x):
    if x>=datetime.time(4, 0, 1) and x <=datetime.time(10, 0, 0):
        return 'morning'
    elif x>=datetime.time(10, 0, 1) and x <=datetime.time(16, 0, 0):
        return 'midday'
    elif x>=datetime.time(16, 0, 1) and x <=datetime.time(22, 0, 0):
        return 'evening'
    elif x>=datetime.time(22, 0, 1) or x <=datetime.time(4, 0, 0):
        return 'late night'
    


# In[67]:


figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
taxi_df['pickup_hour']=taxi_df['pickup_datetime'].dt.hour
taxi_df.pickup_hour.hist(bins=24,ax=ax[0])
ax[0].set_title('Distribution of pickup hours')

taxi_df['dropoff_hour']=taxi_df['dropoff_datetime'].dt.hour
taxi_df.dropoff_hour.hist(bins=24,ax=ax[1])
ax[1].set_title('Distribution of dropoff hours')


# In[ ]:


#The highest trips distance are started in evening (between 16 and 20 hours) 
#and the least are  in the late night and early morning between 3–6 hours)


# In[69]:


taxi_df['store_and_fwd_flag'].value_counts()


# In[70]:


#The number of N flag is much larger. 
#We can later see whether they have any relation with the duration of the trip.


# In[103]:


sns.distplot(taxi_df['trip_distance'],kde=False)
plt.title('The distribution of the PickUpDistancedistribution')


# In[ ]:


#This histogram shows extreme left skewness, hence there are outliers.


# In[6]:


print(taxi_df['trip_distance'].nlargest(10))


# In[81]:


sns.distplot(taxi_df['trip_distance'])
plt.title('Distribution of the pickup distance after the treatment of outliers')


# In[82]:


#Still there is an extreme right skewness. Thus we will divide the trip_duration column into some interval.

#The intervals are decided as follows:

    #less than 5 hours
    #5–10 hours
    #10–15 hours
    #15–20 hours
    # than 20 hours


# In[83]:


bins=np.array([0,1800,3600,5400,7200,90000])
taxi_df['distance_time']=pd.cut(taxi_df.trip_distance,bins,labels=["< 5", "5-10", "10-15","15-20",">20"])


# In[85]:


bins=np.array([0,1800,3600,5400,7200,90000])
taxi_df['distance_time']=pd.cut(taxi_df.trip_distance,bins,labels=["<5","5-10","10-15","15-20",">20"])


# In[86]:


sns.distplot(taxi_df['pickup_longitude'])
plt.title('The distribution of Pick up Longitude')


# In[100]:


sns.distplot(taxi_df['pickup_latitude'])
plt.title('The distribution of Pick up Latitude')


# In[ ]:


sns.distplot(taxi_df['dropoff_latitude'])
plt.title('The distribution of drop off Latitude')


# In[ ]:


sns.distplot(taxi_df['dropoff_longitude'])
plt.title('The distribution of dropoff longitude')


# In[ ]:


taxi_df['vendor_id'].hist(bins=2)


# In[ ]:


#The distribution of vendor id is not much different as expected.


# In[7]:


#The relationship between passenger count and trip distance
sns.relplot(x="passenger_count", y="trip_distance", data=taxi_df, kind="scatter")


# In[10]:


#Here we see, passenger count has no such relationship with trip duration. 
#Thus we see there is only value above 200
#while all the others are somewhere between 0 and 9.
#The one near 200 is definitely an outlier which must be treated.


# In[ ]:


a=sns.relplot(x="passenger_count", y="trip_distance", data=taxi_df, kind="scatter")
a.set(xlim=(1, 9))
plt.show()


# In[ ]:


#The relationship between vendor id and distance


# In[ ]:


b=sns.catplot(x="vendor_id", y="trip_distance",kind="strip",data=taxi.df)
b.set(ylim=(1, 9))


# In[ ]:


#The relationship between geographical location and distance


# In[ ]:


sns.relplot(x="pickup_latitude", y="dropoff_latitude",hue='pickup_timezone',data=taxi_df);


# In[ ]:


#The relationship between store forward flag and distance


# In[ ]:


sns.catplot(x="store_and_fwd_flag", y="trip_distance",kind="strip",data=taxi_df)


# In[ ]:




