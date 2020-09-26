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
import datetime

#load data

data = pd.read_csv('wind_generation_data.csv', sep=",")

# Split data into training and test sets

X = data.drop(['Power Output'], axis = 1).values # X are the input (or independent) variables
y = data['Power Output'].values # Y is output (or dependent) variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lm = linear_model.LinearRegression()
lm.fit(X_train,y_train)


# testing the trained model on new data set

#using the urllib library to get the json and calling json.loads() to convert into a dictionary
json_string = urllib.request.urlopen('https://api.openweathermap.org/data/2.5/onecall?lat=53.556563&lon=8.598084&exclude=hourly,minutely&appid=cd279b25c0fba864032427f14a1dc834').read()
weather1 = json.loads(json_string)

#Let's throw this data into a dataframe so we can take a better look at it
data = pd.DataFrame(weather1['daily'])

#dropping all columns we don't need and renaming columns
data1 = data.drop(['sunrise','sunset','pressure','humidity','dew_point', 'clouds','uvi','pop','rain','sunrise','sunset','feels_like','weather','temp'], axis = 1)
data1= data1.rename(columns = {'dt': 'Date','wind_deg': 'direction', 'wind_speed': 'wind speed'}, inplace = False)

#converting date from Epoch to datetime and creating a column called day
data1["Date"] = pd.to_datetime(data1["Date"],unit = 's')
data1["Day"] = data1["Date"].dt.day


#dropping date column for modelling
data2 = data1.drop(['Date','Day'], axis = 1)

#defining our indipendent variable
X1 = data2.values
# getting our predicted value for test data
y = lm.predict(X1)
#print(X1 , y)

#appending the predicted power output back to the test data.
PredictedPower_Wind = pd.DataFrame(y)

data1['PredictedPower_Wind'] = PredictedPower_Wind
data1 = data1[0:5]
#print(data1)

# Let's develop a logic for the maintenance.csv schedule.
data1['PredictedPower_Wind'] = np.select([data1.Day == 3, data1.Day == 5,data1.Day == 7,data1.Day == 8,data1.Day == 15,data1.Day == 24,data1.Day == 28], 
                                                [0.7*data1.PredictedPower_Wind,0.6*data1.PredictedPower_Wind, 0.5*data1.PredictedPower_Wind,0.45*data1.PredictedPower_Wind,0.55*data1.PredictedPower_Wind,
                                                0.9*data1.PredictedPower_Wind,0.3*data1.PredictedPower_Wind ], default=data1.PredictedPower_Wind)
#print(data1)

pickle.dump(lm,open('modelWind.pkl','wb'))

model= pickle.load(open('modelWind.pkl','rb'))











