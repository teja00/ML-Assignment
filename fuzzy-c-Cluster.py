#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 02:44:33 2020

@author: TejaNagubandi
"""

import numpy as np
import pandas as pd
import random

Data = pd.read_excel("DataSets.xlsx",skip_blank_lines = False,error_bad_lines=False)
data = Data.to_numpy()
m = 2
e = 0.001
n = len(data)
k = 2 

def intial_membership_matrix():
    membership_matrix = list()
    for i in range(n):
        rand_list = [random.random() for i in range(k)]
        summation = sum(rand_list)
        temp_list = [x/summation for x in rand_list]
        membership_matrix.append(temp_list)
    return membership_matrix

te =intial_membership_matrix()