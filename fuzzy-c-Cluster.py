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

Data = pd.read_excel("DataSets.xlsx",skip_blank_lines = False,error_bad_lines=False)
data = Data.to_numpy()
x_value = data[:,0]
y_value = data[:,1]
m = 2
e = 0.001
n = len(data)
k = 2
def initial_membership_matrix(k):
    membership_matrix = list()
    for i in range(n):
        rand_list = [random.random() for i in range(k)]
        summation = sum(rand_list)
        temp_list = [x/summation for x in rand_list]
        membership_matrix.append(temp_list)
    return membership_matrix 
def compute_centers(membership_matrix,k):
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
def find_distances(centers,k):
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
def update_membership_matrix(dist,mem,k):
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

def clustering(k):
    new_u = initial_membership_matrix(k)
    old_u = np.zeros_like(new_u)
    i = 1
    while not np.allclose(new_u,old_u):
        i = i + 1
        cen = compute_centers(new_u,k)
        dis = find_distances(cen,k)
        old_u = new_u
        new_u = update_membership_matrix(dis,new_u,k)
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
    
    
    
    
    
obj = list()
iteration = list()
for k in range(2,13):
    v,u,iterations = clustering(k)
    objective_values = objective_function(v,u,k)
    obj.append(objective_values)
    iteration.append(iterations)
    
ob = obj[0:9]
r_value,least = R_objective_function(ob)
v_min,u_min,it = clustering(least)

c = [x for x in range(2,13)]
fig,ax = plt.subplots()
ax.plot(c, obj, color="red", marker="o")
ax.set_xlabel("Value of C",fontsize=14)
ax.set_ylabel("Value of objectiveFunction",color="red",fontsize=14)
ax2=ax.twinx()
ax2.plot(c,iteration,color="blue",marker="o")
ax2.set_ylabel("No. of Iterations",color="blue",fontsize=14)
plt.show()

plt.scatter(x_value,y_value,c= u_min.argmax(axis=1))





