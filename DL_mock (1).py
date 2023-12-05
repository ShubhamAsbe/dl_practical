#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import math


# In[2]:


def sigmoid(x):
    return 1/(1 + math.exp(-x))


# In[20]:


def mae(y, ypred): 
    return abs(y-ypred)


# In[3]:


def perceptronModel(x, w, b):
    v = np.dot(x, w)
    v += b
    y = sigmoid(v)
    return y


# In[4]:


x = np.array([1.0, 0.5, 0.1])
w1 = np.array([1.2, 0.7, 0.9])
w2 = np.array([-0.4, 0.6, 1.1])
w3 = np.array([0.8, 0.5])
b = 0


# In[5]:


def model(x, w1, w2, b):
    x1 = perceptronModel(x, w1, b)
    x2 = perceptronModel(x, w2, b)
    x3 = np.array([x1, x2])
    x4 = perceptronModel(x3, w3, b)
    return x4


# In[6]:


output = model(x, w1, w2, b)


# In[7]:


print(output)


# In[8]:


y = 0.6
y_pred = output


# In[21]:


loss = mae(y, y_pred)


# In[22]:


print(loss)


# In[ ]:


def gradient_descent(x, y, epochs):
    w1 = 1
    bias = 0
    rate = 0.5
    n = len(x)
    for i in range(epochs):
        weight_sum = w1*x+bias
        y_predict =sigmoid(weight_sum)
        loss = log_loss(y, y_predict)
        
        w1d = (1/n)*np.dot(np.transpose(age)*(y_predict-y))
        bias_d = np.mean(y_pedict-y)
        w1 = w1-rate*w1d
        bias = bias - rate*bias_d
        print(f"Epoch: {i}, w1:{w1}, bias:{bias}, loss:{loss}")
    return w1, bias

