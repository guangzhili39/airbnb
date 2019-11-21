# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:45:21 2019
    Clean version of airbnb property price recommendation model.
    Four models are used to predict property price
    1) linear regression
    2) lasso regulization linear regression
    3) support vector machine regression with linear kernel
    4) knn regression
    we use vortingregression to ensemble them together to provide a more stable
    and solid price recommendation.
    clients want to minimize vacancy rate and maximize price. To solve this two 
    objective problem, we used a divide and conquer technique and ask clients
    to input their vacancy rate acceptance threshold from 0 to 10. From existing 
    listing data, we extract those properties with vacancy rate <= the client vacancy
    threshold, then recommend the price on top of those historical data. 
@author: 19083
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
import pickle
from sklearn.model_selection import train_test_split

#%% some features may not be related to pricing and vacancy, so we drop them
# such as host_name, property name, host_id.
todrop=['host_name','name','host_id']
df = pd.read_excel("listings.xlsx")
df.drop(todrop,inplace=True,axis=1)
df.set_index('id',inplace=True)
#%% get vacancy ratio and classify the property into 10 bucket of vacancies
# actually we can any positive number of buckets
n=10
df.availability_365 = df.availability_365/365.0*n
df['vacancy'] = df.availability_365.apply(int)
#%% find neighbourhod_group unique items and transform category variable to integer
# using dummy functions
dummy1 = pd.get_dummies(df.neighbourhood_group)
dummy3 = pd.get_dummies(df.room_type)
df = pd.concat([df,dummy1,dummy3],axis = 1)
#%% split the dataframe based on vacancy ratio
rates = df.vacancy.unique().tolist()
#%% check missing values and fillna with 0, remove price outlines
# here three sigma rule is used.
df.reviews_per_month.fillna(0,inplace=True)
df = df[np.abs(df.price-df.price.mean()) <= (3*df.price.std())]
#%% for each vacancy bucket, try to use regression model to estimate the coefficents 
# create n number of dataframes

dfx = [df[df['vacancy'] <= x] for x in rates]

rates.sort()
for i in rates:
    df = dfx[i].drop(['neighbourhood_group','neighbourhood','room_type','availability_365','vacancy'],axis = 1)
    df = df[np.abs(df.price - df.price.mean()) <= (3*df.price.std())]
    x = df.drop('price',axis = 1)
    y = df.price
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

#%% applying regression models
# define models
    lasso = Lasso(alpha=0.5)
    knn1 = KNeighborsRegressor() 
    lr = LinearRegression()
    svr = SVR(kernel='linear',gamma='auto')
# fiting models    
    lasso.fit(x_train,y_train)
    svr.fit(x_train,y_train)
    lr.fit(x_train, y_train)
    knn1.fit(x_train,y_train)

#%% model evaluation
    mae_ln = mean_absolute_error(y_train, lr.predict(x_train))
    mae_knn1 = mean_absolute_error(y_train, knn1.predict(x_train))
    mae_svr = mean_absolute_error(y_train, svr.predict(x_train))
    mae_lasso = mean_absolute_error(y_train, lasso.predict(x_train))
    print(" training mae = ",mae_ln,mae_lasso,mae_svr,mae_knn1)
#%% on testing data
    mae_ln = mean_absolute_error(y_test, lr.predict(x_test))
    mae_knn1 = mean_absolute_error(y_test, knn1.predict(x_test))
    mae_svr = mean_absolute_error(y_test, svr.predict(x_test))
    mae_lasso = mean_absolute_error(y_test, lasso.predict(x_test))

    print("testing mae = ",mae_ln,mae_lasso,mae_svr,mae_knn1)
    #%% ensembing models and save it to disk
    models = []
    models.append(('lr',lr))
    models.append(('lasso',lasso))
    models.append(('svr',svr))
    models.append(('knn',knn1))

    model = VotingRegressor(models)
    model.fit(x_train,y_train)
    modelname="model.{}.pkl".format(i)
    pickle.dump(model, open(modelname,'wb'))
#%% Linear regression coeficients to boost knn regression
#    for idx,name in enumerate(x_train):
#        if lasso.coef_[idx] > 0:
#            x_train2 = pd.DataFrame(x_train)
#            x_test2 = pd.DataFrame(x_test)
#            x_train2[name] = x_train2[name]*lasso.coef_[idx]
#            x_test2[name] = x_test2[name]*lasso.coef_[idx]
#    knn2 = KNeighborsRegressor()
#    knn2.fit(x_train2,y_train)
#    train_knn2 = mean_absolute_error(y_train, knn2.predict(x_train))
#    test_knn2 = mean_absolute_error(y_test, knn2.predict(x_test))      
#%% testing example
# Loading model to compare the results and make sure it works

uinput={'id':8888,
       'name':'new house',
       'host_id':2845,
       'host_name':'Jennifer',
       'neighbourhood_group':'Manhattan',
       'neighbourhood':'Midtown',
       'latitude':40.753,
       'longitude':-73.98,
       'room_type':'Entire home/apt',
       'minimum_nights':2,
       'vacancy threshold':2}
minput=[]
minput.append(uinput['latitude'])
minput.append(uinput['longitude'])
minput.append(uinput['minimum_nights'])
minput.extend([1,0,0])
if uinput['neighbourhood_group'] == 'Bronx':
    minput.extend([1,0,0,0,0])
elif uinput['neighbourhood_group'] == 'Brooklyn':
    minput.extend([0,1,0,0,0])
elif uinput['neighbourhood_group'] == 'Manhattan':
    minput.extend([0,0,1,0,0])
elif uinput['neighbourhood_group'] == 'Queens':
    minput.extend([0,0,0,1,0])
else:
    minput.extend([0,0,0,0,1])

if uinput['room_type'] == 'Entire home/apt':
    minput.extend([1,0,0])
elif uinput['room_type'] == 'Private room':
    minput.extend([0,1,0])
else:
    minput.extend([0,0,1])

modelname="model.{}.pkl".format(uinput['vacancy threshold'])
model = pickle.load(open(modelname,'rb'))
pinput = []
pinput.append(minput)
price = model.predict(pinput)
print(price)