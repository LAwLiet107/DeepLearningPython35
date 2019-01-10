#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 01:41:35 2019

@author: xiaoe
"""

import mnist_loader
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# my_network2_momentum test
'''
import my_network2_momentum
net = my_network2_momentum.Network([784, 30, 10], cost = my_network2_momentum.CrossEntropyCost)
net.default_weight_initializer()
#net.large_weight_initializer()
net.SGD(training_data[:50000], 60, 30, 0.5, 0.3,
    lmbda = 5,
    evaluation_data = validation_data,
    #monitor_evaluation_cost = True,
    monitor_evaluation_accuracy = True,
    monitor_training_cost = True,
    #monitor_training_accuracy = True,
    early_stopping_n = 10)
'''
# my_network2_relu test
# best accuracy: 97.30%

import my_network2_relu
net = my_network2_relu.Network([784, 30, 10], cost = my_network2_relu.CrossEntropyCost)
#net.relu_weight_initializer()
net.another_relu_weight_initializer()
net.SGD(training_data[:50000], 100, 30, 0.1, 
    mu = 0.05, lmbda = 12,
    evaluation_data = validation_data,
    #monitor_evaluation_cost = True,
    monitor_evaluation_accuracy = True,
    monitor_training_cost = True,
    monitor_training_accuracy = True,
    early_stopping_n = 10)

# network2 test
'''
import network2
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.default_weight_initializer()
net.SGD(training_data[:1000], 30, 10, 0.5,
    lmbda=0.001,
    evaluation_data=validation_data,
    #monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    #monitor_training_accuracy=True,
    early_stopping_n = 10)
'''

# graph

# training_cost
plt.figure()     
plt.plot(net.traincostvalue, color = "r", label = "network:[784,30,10]")
plt.xlabel("epochs")
plt.ylabel("cost")
plt.title("training cost")
plt.grid(1)
# evaluation_accuracy
plt.figure() 
plt.plot(net.evaluationaccuracyvalue, color = "g", label = "network:[784,30,10]")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("evaluation accuracy")
plt.grid(1)
plt.legend()