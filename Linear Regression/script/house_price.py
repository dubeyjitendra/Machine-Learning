# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# reading the dataset
dataset= pd.read_csv('./house_prize.csv')

print(dataset)

# %matplotlib inline
plt.scatter(dataset.Areaofhouse,dataset.priceofhouse)

# pottted charts on Xlabel as indepenedent variable and Ylablel as Depenendent varible
# %matplotlib inline
plt.xlabel('HouseofArea(square ft)')
plt.ylabel('HouseofPrice(INR)')
plt.scatter(dataset.Areaofhouse,dataset.priceofhouse,color='red',marker='+')

# create a object of LinerReegression model

linear_reg_model= linear_model.LinearRegression()

# Fit the Training Model
linear_reg_model.fit(dataset[['Areaofhouse']],dataset.priceofhouse)

# ready to predict,
# I am going predict 3500 sq ft price

print(linear_reg_model.predict([[3500]]))

# Liner Formula = Y=MX+ B
# Where X-> Slope and B intercept


print(linear_reg_model.coef_)


# to get intercept
print(linear_reg_model.intercept_)



# Formula is 

# Y = mx + b

# m=30.35714286 (this is coef_)
# b=455000       (This is intercept_)
# x=3500          (Input value)

# Now am going to sum of all vlaues to match with predicted values

# Predicted Value for 3500 SQ ft = 561250

#formula through

result = 135.78767123*3500+180616.43835616432
print(result)

# check both are equal values

