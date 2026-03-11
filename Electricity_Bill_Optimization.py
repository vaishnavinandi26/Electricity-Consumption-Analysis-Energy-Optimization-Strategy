#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


# In[2]:


df = pd.read_csv("electricity_bill_uncleaned _dataset.csv")
df


# In[3]:


df.head()
df.shape
df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df = df.drop_duplicates()


# In[6]:


num_cols = ['daily_hours_used', 'power_watt', 'monthly_energy_kwh','monthly_energy_kwh','monthly_cost','cost_per_kwh']
discrete_cols = ['switch_on_count','standby_hours','household_members']


# In[7]:


category=['appliance_type', 'room','usage_type','efficiency_flag','season','energy_rating']
df[category]=df[category].astype('category')
category
df.info()


# In[8]:


df.info()


# In[9]:


for col in category:
    df[col] = df[col].astype(str)
    df[col].fillna('Unknown', inplace=True)


# In[10]:


actual_appliances = df[df['appliance_type'] != 'Unknown']['appliance_type']
appliance_mode = actual_appliances.mode()[0]
print(f"The most frequent appliance (Mode) is: {appliance_mode}")


# In[11]:


#  Fill 'Unknown' values with the Mode
# This replaces only the 'Unknown' strings with the calculated mode
df['appliance_type'] = df['appliance_type'].replace('Unknown', appliance_mode)


# In[12]:


# Verify the change
print("\nNew counts for appliance_type:")
print(df['appliance_type'].value_counts())


# In[13]:


df['household_members'] = df['household_members'].round(0).astype('Int64')


# In[14]:


# Round standby_hours to 0 decimal places
df['standby_hours'] = df['standby_hours'].round(0).astype('Int64')


# In[15]:


df['switch_on_count'] = df['switch_on_count'].round(0).astype('Int64')


# In[16]:


plt.boxplot(df['daily_hours_used'])
plt.title("Daily Hours Used")
plt.show()



# In[17]:


plt.boxplot(df['power_watt'])
plt.title("Power Watt")
plt.show()


# In[18]:


df.loc[df['daily_hours_used'] > 24, 'daily_hours_used'] = 24
df.loc[df['power_watt'] > 5000, 'power_watt'] = 5000
df.loc[df['monthly_energy_kwh'] < 0, 'monthly_energy_kwh'] = 0


# In[19]:


df['monthly_cost'] = df['monthly_energy_kwh'] * df['cost_per_kwh']


# In[20]:


df.isnull().sum()


# In[21]:


# calculate median power for each appliance
power_median = df.groupby('appliance_type')['power_watt'].median()

# fill missing values using map
df['power_watt'] = df['power_watt'].fillna(
    df['appliance_type'].map(power_median)
)


# In[22]:


df['monthly_energy_kwh'] = (df['power_watt']/1000) * df['daily_hours_used'] * 30


# In[23]:


df['monthly_cost'] = df['monthly_energy_kwh'] * df['cost_per_kwh']


# In[24]:


df['efficiency_flag'] = 'Inefficient'

df.loc[
    (df['energy_rating'].isin(['4 Star','5 Star'])) &
    (df['appliance_age_years'] < 5) &
    (df['standby_hours'] < 3),
    'efficiency_flag'
] = 'Efficient'


# In[25]:


df['possible_wastage'] = 'Normal'

df.loc[
    (df['daily_hours_used'] > 8) |
    (df['standby_hours'] > 5) |
    (df['switch_on_count'] > 15) |
    (df['appliance_age_years'] > 8),
    'possible_wastage'
] = 'High Wastage'


# In[26]:


df['usage_type'] = 'Low Usage'

df.loc[df['daily_hours_used'] > 4, 'usage_type'] = 'Moderate Usage'
df.loc[df['daily_hours_used'] > 8, 'usage_type'] = 'High Usage'


# In[27]:


missing_percent = df['appliance_type'].isna().mean() * 100
print(round(missing_percent, 2), "%")


# In[28]:


(df['appliance_type'].value_counts(normalize=True) * 100).round(2)


# In[29]:


df['appliance_type'] = df['appliance_type'].fillna(
    df.groupby('room')['appliance_type'].transform(lambda x: x.mode()[0])
)


# In[30]:


df.info()


# In[31]:


df.isnull().sum()


# In[32]:


# calculate median hours for each appliance
hours_median = df.groupby('appliance_type')['daily_hours_used'].median()

# fill missing values
df['daily_hours_used'] = df['daily_hours_used'].fillna(
    df['appliance_type'].map(hours_median)
)


# In[33]:


df['power_watt'] = df['power_watt'].clip(upper=5000)
df['daily_hours_used'] = df['daily_hours_used'].clip(upper=24)


# In[34]:


num_cols = [col for col in num_cols if col in df.columns]


# In[35]:


df['monthly_energy_kwh'] = df['monthly_energy_kwh'].round(2)
df['appliance_age_years'] = df['appliance_age_years'].round(2)
df['ambient_temperature_c'] = df['ambient_temperature_c'].round(2)


# In[36]:


df[['power_watt','monthly_energy_kwh','monthly_cost']].boxplot(figsize=(10,5))


# In[37]:


df.select_dtypes(include='number').boxplot(figsize=(18,8))


# In[38]:


df


# In[39]:


for col in category:
    unknown_pct = (df[col] == 'Unknown').sum() / len(df) * 100
    print(col, "Unknown %:", round(unknown_pct, 2))


# In[40]:


for col in category:
    print("\nColumn:", col)
    print("Unknown count:", (df[col] == 'Unknown').sum())
    print("NaN count:", df[col].isna().sum())
    print("Total rows:", len(df))


# In[41]:


for col in num_cols:
    unknown_pct = (df[col] == 'Unknown').sum() / len(df) * 100
    print(col, "Unknown %:", round(unknown_pct, 2))


# In[42]:


df.isnull().sum()


# In[43]:


df.info()
    


# In[44]:


df.to_csv("cleaned_electricity_dataset_f1.csv", index=False)


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x="appliance_type", y="monthly_energy_kwh", data=df, palette="bright")
plt.title("Energy Consumption by Appliance")
plt.xticks(rotation=45)
plt.show()


# In[49]:


plt.scatter(df["power_watt"], df["monthly_energy_kwh"], color="purple")
plt.xlabel("Power (Watt)")
plt.ylabel("Monthly Energy (kWh)")
plt.title("Power vs Energy Usage")
plt.show()


# In[51]:


sns.boxplot(x="energy_rating", y="monthly_cost", data=df, palette="bright")
plt.title("Monthly Cost by Energy Rating")
plt.show()


# In[52]:


plt.hist(df["standby_hours"], bins=10,color="darkorange", edgecolor="black")
plt.title("Standby Hours Distribution")
plt.xlabel("Standby Hours")
plt.show()


# In[55]:


import seaborn as sns
import matplotlib.pyplot as plt
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:




