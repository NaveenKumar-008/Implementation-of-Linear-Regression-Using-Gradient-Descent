# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.


2.Write a function computeCost to generate the cost function.


3.Perform iterations og gradient steps with learning rate.


4.Plot the Cost function using Gradient Descent and generate the required graph.

 


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: NAVEEN KUMAR M
RegisterNumber:  212221040113
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta)
def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions -y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))

  return theta,J_history
  theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) *"+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\theta)$")
plt.title("Cost frunction using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.ylabel("Profit predictions")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))




```

## Output:

### Profit Prediction graph
![ex 3 1](https://user-images.githubusercontent.com/128135244/229801432-b06874b8-8da4-463f-9930-a36ba0b115c0.png)

### Compute Cost Value
![ex 3 2](https://user-images.githubusercontent.com/128135244/229801597-ac07577f-3fbe-4bbe-bc5b-309630326de2.png)

### h(x) Value
![ex 3 3 ](https://user-images.githubusercontent.com/128135244/229801691-406135de-cd87-4b42-8739-f94312f0f761.png)

### Cost function using Gradient Descent Graph
![ex 3 4](https://user-images.githubusercontent.com/128135244/229801813-26187c4f-66a6-439e-a146-7496481b7f8f.png)

### Profit Prediction Graph
![ex 3 5](https://user-images.githubusercontent.com/128135244/229801913-210e853c-a37f-45e4-8e29-65529d4743f9.png)

### Profit for the Population 35,000
![ex 3 6](https://user-images.githubusercontent.com/128135244/229802008-4bbb16dc-16c9-4ba5-87f9-7f7cd2c9dca5.png)

### Profit for the Population 70,000
![ex 3 7](https://user-images.githubusercontent.com/128135244/229802090-e9a9809c-fa29-469c-9583-c9f88058aed8.png)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
