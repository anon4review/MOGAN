# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:54:38 2019

@author: Founder
"""

import pandas as pd
import numpy as np

import zdt
import gan_function
import operate

dim = 30

file_read = 'E:/GANs/kdd/5.22/zdt1_init.csv'

data_ori = pd.read_csv(file_read, engine='python')
datap = zdt.pareto(data_ori.values)
xxx = datap.shape[0]
data1 = zdt.function4(gan_function.function(datap[:, :dim]))[:70, :]
print("第一次评价：%d" %(70-xxx))

data2 = operate.function(zdt.pareto(data1)[:, :dim])
data2 = zdt.function4(data2)[:60, :]
print("第二次评价：60")
#
data3 = zdt.function4(gan_function.function(zdt.pareto(data2)[:, :dim]))[:70+xxx, :]
print("第三次评价：%d" %(70+xxx))

data_out = pd.DataFrame(data3)
output = file_read.strip('.csv')   
output = output + '_300.csv'
data_out.to_csv(output, index=False)
