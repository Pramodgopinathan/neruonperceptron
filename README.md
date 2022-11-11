![](https://github.com/Pramodgopinathan/neruonperceptron/blob/e6211e65301623c7fe08a65d2276b7a2c93a9f72/DeepLearning-Lab1Part1.png)

## Simple Neural Network

![](https://github.com/Pramodgopinathan/neruonperceptron/blob/1f31260d67045a96da2417718a7d25e013f3c1c1/neron1.jpg)

```python
# we have two parameter one is to identify y or z which is 

# y = (x1 * W1 ) + (x2 * W2) + (x3 * w3) + bais
# n = 3 (as because there are three input nerons givens)

import numpy as np

# Above example x [0.8, 0.6, 0.4], w [0.1,0.3,-0.2] and bais = 0.35

X = np.array([0.8,0.6,0.4])
W = np.array([0.1,0.3,-0.2])
bais = 0.35

print('--------------------------------------------------------------------------------------------') 
print('------------------------Finding Y output ---------------------------------------------------')
print('--------------------------------------------------------------------------------------------')
Y = np.dot(X,W) + bais
print('The input of a simple neuron Y for the given network is:' , X)
print('The weight of a simple neuron Y for the given network is:' , W)
print('The output of a simple neuron Y for the given network is:' , Y)

# Define in the problem statement 
theta = 0 

def binary_step(y):
      if y < theta:
            return theta
      else:
            return 1

# Define in the problem statement 
theta = 0 

def bipolar_step(y):
      if y < theta:
            return theta
      else:
            return -1

def sigmoid_binary(x):
      return 1/(1 + np.exp(-x))

def sigmoid_bipolar(x):
      return (np.exp(x)-1)/(np.exp(x)+1)

# Activation function = f(Y) 
print('--------------------------------------------------------------------------------------------') 
print('------------------------Activitation Function-----------------------------------------------')
print('--------------------------------------------------------------------------------------------')
binary_step = binary_step(Y)
bipolar_step = bipolar_step(Y)
print('The output of a Y is 0.53 and after coverging with binary step activition function result is:' , binary_step) 
print('The output of a Y is 0.53 and after coverging with bipolar step activition function result is:' , bipolar_step) 
print('The output of a Y is 0.53 and after coverging with sigmoid binary activition function result is:' , sigmoid_binary(Y)) 
print('The output of a Y is 0.53 and after coverging with sigmoid bipolar activition function result is:' , sigmoid_bipolar(Y)) 

```
# Result 

![](https://github.com/Pramodgopinathan/neruonperceptron/blob/1366350cc2279ce06746df7074aaf8eb6788f6e3/outcome.jpg)
