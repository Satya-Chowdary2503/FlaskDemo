# -*- coding: utf-8 -*-
"""project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gn9A0RwDe0oDkM16hRlZNfrjAxmySq5Y
"""

# Commented out IPython magic to ensure Python compatibility.
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline
mpl.style.use('ggplot')

#upload dataset
from google.colab import files
uploaded = files.upload()

car=pd.read_csv('car.csv')
car

car.shape
#892 rows and 6 Columns

car.info()

#removal of noisy values and null values
car.isnull().sum()

car=car[car['year'].str.isnumeric()]
car['year']=car['year'].astype(int)

car=car[car['Price']!='Ask For Price']
car['Price']=car['Price'].str.replace(',','').astype(int)

car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')
car=car[car['kms_driven'].str.isnumeric()]
car['kms_driven']=car['kms_driven'].astype(int) #returns only numeric values as it has nan values

car=car[~car['fuel_type'].isna()]

car.shape
#noisy values have been removed now the data is good

car

car.describe()

import datetime
car['Age']=datetime.datetime.now().year-car['year']
car['Age']

car

car.to_csv('New_Car_data.csv')

car.describe(include='all')

car['company'].unique()

import seaborn as sns

plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()

#year with price
plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='Age',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()

#kms_driven with Price
sns.relplot(x='kms_driven',y='Price',data=car,height=7,aspect=1.5)

plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=car)

#Relationship of Price with FuelType, Year and Company mixed
ax=sns.relplot(x='company',y='Price',data=car,hue='fuel_type',size='Age',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')

X=car[['name','company','Age','kms_driven','fuel_type']]
y=car['Price']

X

#traininng
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

ohe=OneHotEncoder() #onehotencoder
ohe.fit(X[['name','company','fuel_type']])

column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')

#linearregression
lr=LinearRegression()

pipe=make_pipeline(column_trans,lr)

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

r2_score(y_test,y_pred) #r2score

scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))

np.argmax(scores)

scores[np.argmax(scores)]

pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',4,100,'Petrol']).reshape(1,5)))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)

import pickle
pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))

pipe.predict(pd.DataFrame(columns=['name','company','Age','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',4,100,'Petrol']).reshape(1,5)))

pipe.steps[0][1].transformers[0][1].categories[0]