#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:33:45 2019

@author: xiaoe
"""

import json
import random
import sys

import numpy as np

#### Define Cost Function
class CrossEntropyCost(object):
    
    # 静态方法
    @staticmethod
    def fn(a, y):
        """计算代价函数C"""
        return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1 - a)))
    
    @staticmethod
    def delta(a, y):
        """计算delta(L)"""
        return (a-y)
    
#### Main Network Class
class Network(object):
    
    def __init__(self, sizes, cost = CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        self.traincostvalue = []
        self.evaluationaccuracyvalue = []
        
    def default_weight_initializer(self):
        """初始化weights和biases"""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(1/x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.velocities = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
      
    def large_weight_initializer(self):
        """最初的初始化方法"""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.velocities = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def feedforward(self, a):
        """前向传播"""
        for b, w in zip(self.biases, self.weights):
            # z = w * a + b
            a = sigmoid(np.dot(w, a) + b)
        # 返回前向传播的最终结果的一个vector
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, mu, 
            lmbda = 0.0, evaluation_data = None,
            monitor_evaluation_cost = False, monitor_evaluation_accuracy = False,
            monitor_training_cost = False, monitor_training_accuracy = False, 
            early_stopping_n = 0):
        
        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)
            
        # training_data的数据量
        n = len(training_data)
            
        # early stopping functionality
        best_accuracy = 0
        no_accuracy_change = 0
        
        # 分别储存evaluation_data的cost,accuracy和training_data的cost,accuracy
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = [], [], [], []
        
        # 将training_data分割成mini_batch
        for j in range(epochs):
            # 将training_data乱序
            random.shuffle(training_data)
            # 将training_data分割为mini_batch,每个mini_batch中数据量为mini_batch_size,并存在list中
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, mu, lmbda, n)
            
            # 输出当前epochs
            print("Epoch %s training complete" % j)
            # 在使用training_data训练时
            # 将当前training_data的cost加入training_cost
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                self.traincostvalue.append(cost)
                print("Cost on training data: {}".format(cost))
            # 将当前training_data的accuracy加入training_accuracy
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert = True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
                
            # 在使用evaluation_data测试时
            # 将当前evaluation_data的cost加入evaluation_cost
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert = True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            # 将当前evaluation_data的accuracy加入evaluation_accuracy
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                self.evaluationaccuracyvalue.append(accuracy/10000)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
            
            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                else:
                    no_accuracy_change += 1
                # 当accuracy连续降低的次数达到early_stopping_n,则程序提前停止
                if no_accuracy_change == early_stopping_n:
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
        
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
    
    def update_mini_batch(self, mini_batch, eta, mu, lmbda, n):
        """用momentum代替gradient decent"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, lmbda, n)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # momentum
        # v = mu * v - eta * nabla_C
        self.velocities = [mu * v - (eta / len(mini_batch)) * nw for v, nw in zip(self.velocities, nabla_w)]
        # w = (1 - eta * lambda / n) * w + v
        self.weights = [(1 - eta * lmbda / n) * w + v for w, v in zip(self.weights, self.velocities)]
        #self.weights = [w + v for w, v in zip(self.weights, self.velocities)]
        # b = b - eta * nabla_C
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self ,x, y, lmbda, n):
        """backpropagation"""
        nabla_b = [np.shape(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        # input
        activation = x
        # list to store all the activations, layer by layer
        activations = [x]
        # list to store all the z vectors, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            # z = w * a + b
            z = np.dot(w, activation) + b
            # z值加入list
            zs.append(z)
            # a = sigmoid(z)
            activation = sigmoid(z)
            # a值加入list
            activations.append(activation)
        # backward pass
        # delta(L) = a - y
        delta = (self.cost).delta(activations[-1], y)
        # nabla_b(L) = delta(L)
        nabla_b[-1] = delta
        # nabla_w(L) = delta(L) * a(L-1).transpose()
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 从倒数第二层开始
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def accuracy(self, data, convert = False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. """
        if convert:
            # training data
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            # validation or test data
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        result_accuracy = sum(int(x == y) for (x, y) in results)
        # 返回本次训练/测试回合的正确率
        return result_accuracy
        
    def total_cost(self, data, lmbda, convert = False):
        """Return the total cost for the data set ``data``.The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data. """
        cost = 0.0
        for x, y in data:
            # a为前向传播最终结果的vector
            a = self.feedforward(x)
            if convert:
                # validation or test data
                y = vectorized_result(y)
            # 未正则化的cost
            cost += self.cost.fn(a, y) / len(data)
            # 正则化项
            cost += 1/2 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        # 返回本次训练/测试回合的总cost
        return cost

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
            
def sigmoid(z):
    """the sigmoid function"""
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """sigmoid的导数"""
    return sigmoid(z) * (1 - sigmoid(z))