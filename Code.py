import pandas as pd 
import numpy as np 
from sklearn import linear_model
import pickle 
from sklearn import joblib 

data = pd.read_csv("C:/Users/Chaitanya/Downloads/archive (6).zip")

data.bedrooms.median()
data.bedrooms = data.bedrooms.fillna(data.bedrooms.median())
reg = linear_model.LinearRegression()
reg.fit(data[['area', 'bedrooms', 'age']], data.price)

with open('reg.pickle', 'wb') as f: 
    pickle.dump(reg, f)

with open('reg.pickle', 'rb') as f:
    mp = pickle.load(f)
    
   
    
prediction = reg.predict([[3000, 3, 15]])
prediction1 = reg.predict([[3400, 4, 14]])
prediction2 = mp.predict([[5990, 4, 10]])



joblib.dump(reg, 'reg_joblib')
joblib.load('model_joblib')
