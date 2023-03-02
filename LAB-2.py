#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np #importing library numpy
import pandas as pd #importing library pandas
import matplotlib.pyplot as plt #importing library matplotlib.pyplot for plotting purposes
from sklearn.linear_model import LinearRegression  #importing linear regression

data=pd.read_csv('ex1data1.txt',header=None) #to read a file using  the python library pandas,if header =None is not given then the first value wil get into header
plt.scatter(data[0],data[1]) #for plotting the given data 
data_n=data.values  #for making the data into arrays
m=len(data_n[:,0])  #to find the number of data given[0 deonotes first coloumn]
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1) #to make the matrix multiplication easier we are adding '1' infront of every first element
y=data_n[:,1].reshape(m,1) #for changing the shape of y elements inorder to make matrix multiplication possible

regressor=LinearRegression()  # linear regression is assigned  to the variable regressor
regressor.fit(X,y)  #method estimates the coefficients of the linear regression model so that the sum of the squared differences between the predicted and actual values is minimum.

theta0=regressor.intercept_ #assigns the intercept term of the linear regression model to the variable theta0.
theta1=regressor.coef_   #assigns the slope term of the linear regression model to the variable theta1.

plt.plot(X,theta1*X+theta0) #for plotting the model
plt.show()#Display


# In[ ]:




