# # Logistic Regression With Binary Classification
# 
# Logistic Binary classification has one independent variable and one dependent variable, where as dependent variable is either "0"or "1" or True" or "False"
# 
# Logit or Sigmoid formula: 
# 
# 
# f(x)=  1/ 1 + e(power(-z)) 
# 
# where z= mx + b


# Import all necessary packages

import pandas as pd
import numpy as np
from matplotlib import pyplot


# import scikit library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Reading dataset into pandas dataframe
df =pd.read_csv("buy_insurance.csv")


# check dataset records
df.head()


# if you want to check last five records
df.tail()


# Represent complete dataset in graphical form
pyplot.scatter(df.limit_age,df.purchased_insurance, color='red',marker='*')


# spliting dataset in training tessting forms
X_train,X_test,y_train,y_test =train_test_split(df[["limit_age"]],df.purchased_insurance, random_state=42, train_size=0.20)



# Initilize the objct
log_model = LogisticRegression()


# Fitting the model
ft_logistic_model = log_model.fit(X_train,y_train)


# predicating single records
ft_logistic_model.predict(20)


# Predicating all testing set
ft_logistic_model.predict(X_test)


# Generating overall score 
result = round(ft_logistic_model.score(X_test,y_test),2)
print(result)

