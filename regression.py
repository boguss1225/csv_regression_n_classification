import pandas as pd
import numpy as np
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import normalize,LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime

# Define data refactorier
def discretize_date(current_date, t):
    current_date = current_date[:-10]
    cdate = datetime.strptime(current_date, '%Y-%m-%d %H:%M:%S')

    if t == 'hour_sin':
        return np.sin(2 * np.pi * cdate.hour/24.0)
    if t == 'hour_cos':
        return np.cos(2 * np.pi * cdate.hour/24.0)
    if t == 'day_sin':
        return np.sin(2 * np.pi * cdate.timetuple().tm_yday/365.0)
    if t == 'day_cos':
        return np.cos(2 * np.pi * cdate.timetuple().tm_yday/365.0)

# Input Data
print("Import Data")
df=pd.read_csv("weather_dataset/weatherHistory.csv")
print(df.head())
print(df.info())
print(df.nunique())

# Drop unnecessary columns
df.drop(['Daily Summary'],axis=1,inplace=True)
df.dropna(inplace=True) # drop null values

# Refactory Date colum
date_types = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
for dt in date_types:
    df[dt] = df['Formatted Date'].apply(lambda x : discretize_date(x, dt))
df.drop(['Formatted Date'],axis=1,inplace=True)

# Convert categorical data into numerical data
le = LabelEncoder()
df['Summary']=le.fit_transform(df['Summary'])
df['Precip Type']=le.fit_transform(df['Precip Type'])

# category variable
category_var=["Summary",'Precip Type']
category_df=df[category_var]

# Continuous variable
continuous_var=["Temperature (C)","Apparent Temperature (C)","Humidity","Wind Speed (km/h)","Wind Bearing (degrees)","Visibility (km)","Loud Cover","Pressure (millibars)"]
continuous_df=df[continuous_var]

# encoding the categorical columns
df = pd.get_dummies(df, columns = category_var, drop_first = True)

# normalize the continuous data
df[continuous_var]=normalize(df[continuous_var])

# extract the labels from the data
y=df['Temperature (C)']
X=df.drop('Temperature (C)', axis=1)

# divide into train and 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)

reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
models,predictions = reg.fit(X_train, X_test, y_train, y_test)

# result
print("regression prediction of temperature")
print(models)
print("done")
