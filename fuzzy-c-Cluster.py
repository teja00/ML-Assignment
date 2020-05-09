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
import operator

Data = pd.read_excel("DataSets.xlsx",skip_blank_lines = False,error_bad_lines=False)
data = Data.to_numpy()
m = 2
e = 0.001
n = len(data)
k = 2 


te = initial_membership_matrix()
ce = compute_centers(te)
d = find_distances(ce)

def initial_membership_matrix():
    membership_matrix = list()
    for i in range(n):
        rand_list = [random.random() for i in range(k)]
        summation = sum(rand_list)
        temp_list = [x/summation for x in rand_list]
        membership_matrix.append(temp_list)
    return membership_matrix 
def compute_centers(membership_matrix):
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
def find_distances(centers):
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
def update_membership_matrix(dist,mem):
    dist = np.transpose(dist)
    mem = np.array(mem)
    for i in range(n):
        for j in range(k):
            if dist[i][j] > 0:
                val = dist[i][j]
                su = 0
                for l in range(k):
                    su = su + (math.pow((val/dist[i][l]),(2/m-1)))
                mem[i][j] = 1/su
            else:
                mem[i][j] = 1
                for s in range(k):
                    if(s != j):
                        mem[i][s] = 0              
    return mem   

    
    

    
    



        