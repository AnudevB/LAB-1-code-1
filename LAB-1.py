#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np  #importing library numpy 
import pandas as pd  #importing library pandas
import matplotlib.pyplot as plt  #importing library matplotlib.pyplot for plotting purposes


def Fn_Cost(X,y,theta): #defining a function with values X,y,theta
    h=X.dot(theta)  #dot product calculates the predicted value 
    square_err=(h-y)**2  #calculating the squared error between the predicted and the actual values
    return 1/(2*m)*np.sum(square_err)  # calculates the mean squared error 


def G_D(X,y,theta,alpha,num_iters):
    J_history=[]  #creating a list
    for i in  range(num_iters):
        h=X.dot(theta)  #dot product calculates the predicted value
        error=np.dot(X.transpose(),(h-y))  #calculates the dot product of the transpose of X and the  errors
        descent=alpha*(1/m)*error  #it calculates the amount by which the model parameters should be updated to minimize the cost function(alpha is the learning rate,m is the number of samples)
        theta-=descent   #this means theta=theta-descent,which corresponds to minimizing the cost function.
        J_history.append(Fn_Cost(X,y,theta)) #is an expression that appends the current value of the cost function to a list
    return theta,J_history #returns the values


data=pd.read_csv('ex1data1.txt',header=None) #to read a file using  the python library pandas,if header =None is not given then the first value wil get into header
data_n=data.values     #for making the data into arrays
m=len(data_n[:,0])    #to find the number of data given[0 deonotes first coloumn]
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)    #to make the matrix multiplication easier we are adding '1' infront of every first element
y=data_n[:,1].reshape(m,1)    #for changing the shape of y elements inorder to make matrix multiplication possible
theta=np.zeros((2,1))    # creates an array of shape (2,1) filled with zeros
Fn_Cost(X,y,theta)    #calling the function Fn_cost with parameters X,y,theta
theta,J_history=G_D(X,y,theta,0.01,1500)  #calling the function G_D and returning the results into theta and J_histroy
theta[0,0] #corresponds to the intercept term of the linear model
plt.scatter(data[0],data[1])#for plotting the given data 
plt.plot(X,theta[1,0]*X+theta[0,0])#for plotting the model
plt.show()#Display


# In[ ]:




