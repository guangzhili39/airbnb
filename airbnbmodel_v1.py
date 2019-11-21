# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:45:21 2019

@author: 19083
"""
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
#%% some features may not be related to pricing and vacancy, so we drop them
todrop=['host_name','name','host_id']
df = pd.read_excel("listings.xlsx")
df.drop(todrop,inplace=True,axis=1)
df.set_index('id',inplace=True)
#%% get vacancy ratio and classify the property into 10 bucket of vacancies
train_result = []
test_result = []

df.availability_365 = df.availability_365/365.0*10
df['vacancy'] = df.availability_365.apply(int)
#%% find neighbourhod_group unique items and transform category variable to integer
# using dummy functions
dummy1 = pd.get_dummies(df.neighbourhood_group)
#dummy2 = pd.get_dummies(df.neighbourhood)
dummy3 = pd.get_dummies(df.room_type)
df = pd.concat([df,dummy1,dummy3],axis = 1)
#%% split the dataframe based on vacancy ratio
rates = df.vacancy.unique().tolist()
#%% check missing values and fillna with 0, remove price outlines
df.reviews_per_month.fillna(0,inplace=True)
# import seaborn as sns
# sns.boxplot(x=df['price'])
df = df[np.abs(df.price-df.price.mean()) <= (3*df.price.std())]
#%% for each vacancy bucket, try to use regression model to estimate the coefficents 
#df.corr()['price'].sort_values()

dfx = [df[df['vacancy'] <= x] for x in rates]
from sklearn.model_selection import train_test_split
mae1_lasso=0
mae2_lasso=0
mae1_ln = 0
mae2_ln = 0
mae1_svr = 0
mae2_svr = 0
mae1_knn = 0
mae2_knn = 0
rates.sort()
for i in rates:
    df = dfx[i].drop(['neighbourhood_group','neighbourhood','room_type','availability_365','vacancy'],axis = 1)
    df = df[np.abs(df.price - df.price.mean()) <= (3*df.price.std())]
    x = df.drop('price',axis = 1)
    y = df.price
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
#%% import scale
#    from sklearn.preprocessing import StandardScaler
#    scaler = StandardScaler()
#    train_scaled = scaler.fit_transform(x_train)
#    test_scaled = scaler.transform(x_test)
#    train_scaled = x_train
#    test_scaled = x_test
#%% learner regression
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import Lasso

    lasso = Lasso(alpha=0.5)
    lasso.fit(x_train,y_train)
    
    knn1 = KNeighborsRegressor() 
    model = LinearRegression()
    svr = SVR(kernel='linear',gamma='auto')
    svr.fit(x_train,y_train)
    model.fit(x_train, y_train)
    knn1.fit(x_train,y_train)
#%% view linear regression coefficients    
#    print(model.coef_)
#    print(model.intercept_)
#    print(lasso.coef_)
#    print(lasso.intercept_)

#    for idx,name in enumerate(x_train):
#        if lasso.coef_[idx] > 0:
#            x_train2 = pd.DataFrame(x_train)
#            x_test2 = pd.DataFrame(x_test)
#            x_train2[name] = x_train2[name]*lasso.coef_[idx]
#            x_test2[name] = x_test2[name]*lasso.coef_[idx]
#    knn2.fit(x_train,y_train)
#%% model evaluation
#    print("vacancy in ",i,"bucket")
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

 #   mse = mean_squared_error(y_train, model.predict(train_scaled))
    mae_ln = mean_absolute_error(y_train, model.predict(x_train))
    mae_knn1 = mean_absolute_error(y_train, knn1.predict(x_train))
    mae_svr = mean_absolute_error(y_train, svr.predict(x_train))
    mae_lasso = mean_absolute_error(y_train, lasso.predict(x_train))
    mae1_ln += mae_ln
    mae1_svr += mae_svr
    mae1_knn += mae_knn1
    mae1_lasso += mae_lasso
    temp = []
    temp.append(mae_ln)
    temp.append(mae_lasso)
    temp.append(mae_svr)
    temp.append(mae_knn1)
    train_result.append(temp)
    print(" mae = ",mae_ln,mae_lasso,mae_svr,mae_knn1)
#%% on testing data
#    test_mse = mean_squared_error(y_test, model.predict(test_scaled))
    mae_ln = mean_absolute_error(y_test, model.predict(x_test))
    mae_knn1 = mean_absolute_error(y_test, knn1.predict(x_test))
    mae_svr = mean_absolute_error(y_test, svr.predict(x_test))
    mae_lasso = mean_absolute_error(y_test, lasso.predict(x_test))
    
    mae2_svr += mae_svr
    mae2_ln += mae_ln
    mae2_knn += mae_knn1
    mae2_lasso += mae_lasso
    temp = []
    temp.append(mae_ln)
    temp.append(mae_lasso)
    temp.append(mae_svr)
    temp.append(mae_knn1)
    test_result.append(temp)
    print("mae = ",mae_ln,mae_lasso,mae_svr,mae_knn1)
#%% Linear regression coeficients to boost knn regression
    for idx,name in enumerate(x_train):
        if lasso.coef_[idx] > 0:
            x_train2 = pd.DataFrame(x_train)
            x_test2 = pd.DataFrame(x_test)
            x_train2[name] = x_train2[name]*lasso.coef_[idx]
            x_test2[name] = x_test2[name]*lasso.coef_[idx]
    knn2 = KNeighborsRegressor()
    knn2.fit(x_train2,y_train)
    train_knn2 = mean_absolute_error(y_train, knn2.predict(x_train))
    test_knn2 = mean_absolute_error(y_test, knn2.predict(x_test))   
    print("i=",i)
    train_result[i].append(train_knn2)
    test_result[i].append(test_knn2)    
#%%
ln_mae1 = mae1_ln/len(rates)
ln_mae2 = mae2_ln/len(rates)
lasso_mae1 = mae1_lasso/len(rates)
lasso_mae2 = mae2_lasso/len(rates)
knn_mae1 = mae1_knn/len(rates)
knn_mae2 = mae2_knn/len(rates)
svr_mae1 = mae1_svr/len(rates)
svr_mae2 = mae2_svr/len(rates)
print("knn train mae =",knn_mae1, "knn test mae =",knn_mae2)
print("lr train mae = ",ln_mae1, "lr test mae =", ln_mae2)
print("svr train mae = ",svr_mae1, "svr test mae =", svr_mae2)
print("lass train mae = ",lasso_mae1, "lasso test mae =", lasso_mae2)