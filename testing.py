#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:23:12 2020

@author: TejaNagubandi
"""

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

Data = pd.read_excel("mining.xlsx",header = None)
data_y = Data.to_numpy()
Center = pd.read_excel("centers.xlsx",header =None)
v_min = Center.to_numpy() 
least = 0
for i in v_min:
    least = least + 1



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

length = len(data_y)
final_mem = initial_membership_matrix(least,length)
distance = find_distances(v_min,least,data_y,length)
final_u = update_membership_matrix(distance,final_mem,least,length)
k_value = data_y[:,0]
l_value = data_y[:,1]
plt.scatter(k_value,l_value,c=final_u.argmax(axis=1))
plt.show()
