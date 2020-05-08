#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 02:44:33 2020

@author: TejaNagubandi
"""

import numpy as np
import pandas as pd

Data = pd.read_excel("Data Sets.xlsx",index = True,Header=True)
data = Data.to_numpy()
