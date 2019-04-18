# #### This is a Multi Regression Problems
# 
# In defintion terms, when we want to predict the value of a variable based on the value of two or more other variables,
# 
# Independent variable:
# 
# 1-Areaofhouse
# 
# 2-bedrooms
# 
# 3-age
# 
# 
# 
# Dependent variable:
# 
# 1-priceofhouse
# 
# 
# Based on Independent variable we will predict the dependent variable.
# 
# Set the value in multi linear Formula:
# 
# Y = M1X1 + M2X2  + M3X3 + B 
# 
# priceofhouse = m1 * Areaofhouse + m2 * bedrooms + m3 * age + b
# 
# where 
# 
#    m1=m2=m3 is coefficient or called slope.
#    
#    b is intercept


#Importing all neccessary modules 
import pandas as pd
import numpy as np
from sklearn import linear_model


# importing the training dataset

df = pd .read_csv("house_prize_multivarite.csv")

print(df.head())



# Create the object of linear model
multi_lin_model = linear_model.LinearRegression()


# Fitting the training data
multi_lin_model.fit(df[['Areaofhouse','bedrooms','age']],df.priceofhouse)


# Predict the priceofhouse based on input parameter( AreofHouse, bedrooms and age)
multi_lin_model.predict([[1600,4,25]])


# Cofficients

print(multi_lin_model.coef_)


# To get the interceept value
print(multi_lin_model.intercept_)


# Summarizing with Formula

# priceofhouse   = m1 * Areaofhouse + m2 * bedrooms + m3 * age + b

m1=28.67822626

m2 = 18591.65848364

m3 = 528.96790532

Areaofhouse= 1600

bedrooms= 4

age= 25

b= 386681.22383585706

result = m1 * Areaofhouse + m2 * bedrooms + m3 * age + b
print(result) # If we can check both predicated value and summarize is same.





