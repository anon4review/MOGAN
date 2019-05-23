# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:34:41 2018

@author: Founder
"""

import math
import numpy as np
import copy


def pareto(x):
    x = x.tolist()
    l = copy.copy(x)
    list_del = []
    for i in range(len(l)):
        for j in x:
            if j[len(j)-2] < l[i][len(j)-2] and j[len(j)-1] < l[i][len(j)-1]:
                list_del.append(i)
                break
    
    del x, i, j
    list_del.reverse()
    for i in list_del:
        l.pop(i)

    y = sorted(l, key=lambda l:l[len(l)-2])
    return np.array(y)
    

def dectobin_ind(x, r):
    # 十进制转二进制，列表存储
    bin_x = bin(int(x)).lstrip("-")[2:]
    y = []
    for i in range(r):
        if bin_x:
            y.append(int(bin_x[-1]))
            bin_x = bin_x[:-1]
        else:
            y.append(0)
    y.reverse()    
    return y


def dectobin_hang(x):
    x = np.asarray(x)
    lie = []
    for j in range(x.shape[0]):
        if j != 0:
            lie.append(dectobin_ind(x[j], 5))
        else:
            lie.append(dectobin_ind(x[j], 30))
        
    return lie
    

def dectobin(x):
    # 十进制矩阵转二进制
    x = np.asarray(x)
    y = []
    for i in range(x.shape[0]):
        lie = []
        for j in range(x.shape[1]):
            if j != 0:
                lie.append(dectobin_ind(x[i][j], 5))
            else:
                lie.append(dectobin_ind(x[i][j], 30))
        y.append(lie)
    
    return y

    
def judge_0(x):
    for i in x:
        if i < 0 or i > 1:
            return False
    return True

def judge_5(x):
    for i in x:
        if i < -5 or i > 5:
            return False
    return True


def choose_legal(x):
    x = x.tolist()
    l = []
    for i in x:
        if judge_0(i):
            l.append(i)
            
    return l


def concat(x, f1, f2):
    "输入array格式的x和两个数字"
    x = x.tolist()
    x.append(f1)
    x.append(f2)
    "返回拼接的list"
    return x
    

def function1(n, choose=True):
    p = []
    for i in range(n.shape[0]):
        x = n[i]
        if (judge_0(x)):
            f1 = x[0]
            g = 1.0 + 9.0 * (sum(x)-x[0]) / (n.shape[1]-1)
            f2 = g * (1.0 - math.sqrt(x[0]/g))
            if choose:
                p.append(concat(x, f1, f2))
            elif not choose:
                p.append([f1, f2]) 
        
    return np.array(p)
    
    
def function2(n, choose=True):
    p = []
    for i in range(n.shape[0]):
        x = n[i]
        if (judge_0(x)):
            f1 = x[0]
            g = 1.0 + 9.0 * (sum(x)-x[0]) / (n.shape[1]-1)
            f2 = g * (1.0 - (x[0]/g) ** 2)
            if choose:
                p.append(concat(x, f1, f2))
            elif not choose:
                p.append([f1, f2])
        
    return np.array(p)


def function3(n, choose=True):
    p = []
    for i in range(n.shape[0]):
        x = n[i]
        if (judge_0(x)):
            f1 = x[0]
            g = 1.0 + 9.0 * (sum(x)-x[0]) / 29.0
            f2 = g * (1.0 - math.sqrt(x[0]/g) - x[0]/g*math.sin(10*math.pi*x[0]))
            if choose:
                p.append(concat(x, f1, f2))
            elif not choose:
                p.append([f1, f2])
        
    return np.array(p)


def function4(n, choose=True):
    p = []
    for i in range(n.shape[0]):
        x = n[i]
        if x[0]>=0 and x[0]<=1 and judge_5(x[1:]):
            f1 = x[0]
            sum_x = 10*math.cos(4*math.pi*x[0]) - x[0]**2
            for j in x:
                sum_x = sum_x + j**2 - 10*math.cos(4*math.pi*j)
            g = 1.0 + 290.0 + sum_x
            f2 = g * (1.0 - (x[0]/g) ** 2)
            if choose:
                p.append(concat(x, f1, f2))
            elif not choose:
                p.append([f1, f2])
        
    return np.array(p)


def u(z):
    return sum(z)
    
def v(z):
    if u(z) < 5:
        return 2 + u(z)
    else:
        return 1

def function5(nn, choose=True):    
    
    n = dectobin(nn)
    p = []
    for i in range(len(n)):
        x = n[i]
        g = sum(v(x[j]) for j in range(1, len(x)))
        f1 = 1 + u(x[0])
        f2 = g * (1.0 - (f1/g) ** 2)
        if choose:
            p.append(concat(nn[i], f1, f2))
        elif not choose:
            p.append([f1, f2])
        
    return np.mat(p)

def zdt5(nn):
    print(nn)
    x = dectobin_hang(nn)
    g = sum(v(x[j]) for j in range(1, len(x)))
    f1 = 1 + u(x[0])
    f2 = g * (1.0 - (f1/g) ** 2)
    return f1, f2


def function6(n, choose=True):
    p = []
    for i in range(n.shape[0]):
        x = n[i]
        if (judge_0(x)):
            f1 = 1 - (math.exp(-4*x[0]) * ((math.sin(6*math.pi*x[0])) ** 6))
            g = 1.0 + 9.0 * (((sum(x)-x[0]) / 29.0) ** 0.25)
            f2 = 1 - (f1/g) ** 2
            if choose:
                p.append(concat(x, f1, f2))
            elif not choose:
                p.append([f1, f2])
        
    return np.array(p)
