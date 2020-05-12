#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 02:44:33 2020

@author: TejaNagubandi
"""

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

Data = pd.read_excel("Assignment.xlsx",header = None)
data = Data.to_numpy()
data_train = list()
for i in range(600):
    data_train.append(data[i])
data_test_end = list()
for j in range(600,620):
    data_test_end.append(data[i])
data_train = np.array(data_train)
data_test_end = np.array(data_test_end)

data_train1 = list()
data_neglect = data_train[::6]
for i in range(600):
    if(data_train[i] not in data_neglect):
        data_train1.append(data_train[i])
data_train = np.array(data_train1)
data_test = np.concatenate((data_neglect, data_test_end), axis=0)

data_train_dataframe = pd.DataFrame(data_train)
data_train_dataframe.to_excel('classification.xlsx',index = False,header =None)

data_test_dataframe = pd.DataFrame(data_test)
data_test_dataframe.to_excel('mining.xlsx',index = False,header = None)

data_y = data_test
data_x = data
#data_x, data_y = data[:train_pct_index], data[train_pct_index:]

x_value = data_x[:,0]
y_value = data_x[:,1]

m = 2
e = 0.001

def initial_membership_matrix(k,n):
    membership_matrix = list()
    for i in range(n):
        rand_list = [random.random() for i in range(k)]
        summation = sum(rand_list)
        temp_list = [x/summation for x in rand_list]
        membership_matrix.append(temp_list)
    return membership_matrix 
def compute_centers(membership_matrix,k,data,n):
    v= list()
    for i in range(k):
        u = list()
        for j in range(n):
            u.append(membership_matrix[j][i]**m)
        denominator = sum(u)
        numerator = np.dot(u,data)
        x = [x/denominator for x in numerator]
        v.append(x)
    return v
def find_distances(centers,k,data,n):
    Dist = list()
    for i in range(k):
        li = list()
        for j in range(n):
            arr = np.subtract(data[j],centers[i])
            arr1 = np.transpose(arr)
            dist = np.dot(arr1,arr)
            dis = np.sqrt(dist)
            li.append(dis)
        Dist.append(li)
    return Dist
def update_membership_matrix(dist,mem,k,n):
    dist = np.transpose(dist)
    mem = np.array(mem)
    p = float(2/(m-1))
    for i in range(n):
        for j in range(k):
            if dist[i][j] > 0:
                val = dist[i][j]
                den = sum([math.pow(float(val/dist[i][c]), p) for c in range(k)])
                mem[i][j] = float(1/den)       
            elif dist[i][j] == 0:
                mem[i][j] = 1
                for s in range(k):
                    if(s != j):
                        mem[i][s] = 0              
    return mem 

def clustering(k,n):
    new_u = initial_membership_matrix(k,n)
    old_u = np.zeros_like(new_u)
    i = 1
    while not np.allclose(new_u,old_u):
        i = i + 1
        cen = compute_centers(new_u,k,data_x,n)
        dis = find_distances(cen,k,data_x,n)
        old_u = new_u
        new_u = update_membership_matrix(dis,new_u,k,n)
    return cen,new_u,i


def objective_function(v,u,k):
    value = 0
    for i in range(k):
        for j in range(n):
            val = np.subtract(data[j],v[i])
            value = value + ((u[j][i])**m)*(np.dot(np.transpose(val),val))
    return value

def R_objective_function(obj):
    r = list()
    le = len(obj)
    for j in range(1,le-1):
        r.append(abs((obj[j] - obj[j+1])/(obj[j-1] - obj[j])))
    m = 3
    mini = min(r)
    for j in r:
        if(mini == j):
            va = m
            break
        m = m + 1
    return r,va




n = len(data_x)
obj = list()
iteration = list()
for k in range(2,13):
    v,u,iterations = clustering(k,n)
    objective_values = objective_function(v,u,k)
    obj.append(objective_values)
    iteration.append(iterations)
    
ob = obj[0:9]
r_value,least = R_objective_function(ob)
v_min,u_min,it = clustering(least,n)
centers_dataframe = pd.DataFrame(v_min)
centers_dataframe.to_excel('centers.xlsx',index = False,header = None)


c = [x for x in range(2,13)]
fig,ax = plt.subplots()
ax.plot(c, obj, color="red", marker="o")
ax.set_xlabel("Value of C",fontsize=14)
ax.set_ylabel("Value of objectiveFunction",color="red",fontsize=14)
ax2=ax.twinx()
ax2.plot(c,iteration,color="blue",marker="o")
ax2.set_ylabel("No. of Iterations",color="blue",fontsize=14)

plt.show()

plt.scatter(x_value,y_value,c=u_min.argmax(axis=1))
plt.show()


"""
print("this is after the data has been trained to get the centers ")
length = len(data_y)
final_mem = initial_membership_matrix(least,length)
distance = find_distances(v_min,least,data_y,length)
final_u = update_membership_matrix(distance,final_mem,least,length)
k_value = data_y[:,0]
l_value = data_y[:,1]
plt.scatter(k_value,l_value,c=final_u.argmax(axis=1))
plt.show()
"""

