import json
import urllib.request, urllib.parse, urllib.error 
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn import linear_model # for linear regression modeling
from sklearn import preprocessing # for preprocessing like imputting missing values
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import math

data = pd.read_csv('solar_generation_data.csv', sep=",")

#removing degree symbols from temperature
data['Temp Hi'] = data['Temp Hi'].replace('°','', regex=True)
data['Temp Low'] = data['Temp Low'].replace('°','', regex=True)

#filling missing values with median
data.fillna(data.median(), inplace=True)

#converting temperature to float
data['Temp Hi'] = data['Temp Hi'].astype(float)
data['Temp Low'] = data['Temp Low'].astype(float)


# Split data into training and test sets

X = data.drop(['Power Generated in MW', 'Month ', 'Day','Rainfall in mm'], axis = 1).values # X are the input (or independent) variables
y = data['Power Generated in MW'].values # Y is output (or dependent) variable


# create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)


#testing the trained model on new dataset from Openweatherapi


#using the urllib library to get the json and calling json.loads() to convert into a dictionary
import urllib.request

json_string = urllib.request.urlopen('https://api.openweathermap.org/data/2.5/onecall?lat=-20.432510&lon=142.279680&exclude=hourly,minutely&appid=cd279b25c0fba864032427f14a1dc834').read()
weather2 = json.loads(json_string)

#Let's throw this data into a dataframe so we can take a better look at it
df = pd.DataFrame(weather2['daily'])


#Splitting temp dict into individual columns of min and max temperature
df1= pd.DataFrame(df['temp'].values.tolist(), index=df.index)


#dropping all columns we don't need
df2 = df.drop(['sunrise','sunset','pressure','humidity','dew_point', 'wind_speed','wind_deg','pop','sunrise','sunset','feels_like','weather','temp'], axis = 1)

#merging the min and max temp columns onto the dataframe.
df2['Temp Low'] = df1['min']
df2['Temp Hi'] = df1['max']

df2.fillna(df2.median(), inplace=True)


# Rename the columns of df2: df2_new
df2_new = df2.rename(columns = {'dt': 'Date','clouds': 'Cloud Cover Percentage', 'uvi': 'Solar'}, inplace = False)
df2_new["Date"] = pd.to_datetime(df2_new["Date"],unit = 's')
df2_new["Day"] = df2_new["Date"].dt.day

#Let's drop the 'Date' and 'Day' columns and store the data in a new dataframe for doing predictions

df3 = df2_new.drop(['Date','Day'], axis = 1)

#print(df3)

#defining our indipendent variable
X1 = df3.values
# getting our predicted value for test data
y = lm.predict(X1)
#print(X1 , y)

#appending the predicted power output back to the test data.
PredictedPower_Solar = pd.DataFrame(y)

df2_new['PredictedPower_Solar'] = PredictedPower_Solar
df2_new = df2_new[0:5]

#print(df2_new)
# Let's develop a logic the maintenance.csv schedule.
df2_new['PredictedPower_Solar'] = np.select([df2_new.Day == 4, df2_new.Day == 6,df2_new.Day == 19,df2_new.Day == 23,df2_new.Day == 24,df2_new.Day == 25,df2_new.Day == 28], 
                                                [0.03*df2_new.PredictedPower_Solar,0.05*df2_new.PredictedPower_Solar, 0.02*df2_new.PredictedPower_Solar,0.5*df2_new.PredictedPower_Solar,0.2*df2_new.PredictedPower_Solar,
                                                0.05*df2_new.PredictedPower_Solar,0.1*df2_new.PredictedPower_Solar ], default=df2_new.PredictedPower_Solar)


#print(df2_new)


pickle.dump(lm,open('modelSolar.pkl','wb'))

model= pickle.load(open('modelSolar.pkl','rb'))



