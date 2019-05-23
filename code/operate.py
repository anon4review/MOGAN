# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:05:21 2019
operation for GAN
@author: Founder
"""

import numpy as np
import math


c_rate = 0.8
m_rate = 0.2

ub = [1 for i in range(30)]
lb = [0 for i in range(30)]

ub_5 = [5 for i in range(30)]
lb_5 = [-5 for i in range(30)]
ub_5[0] = 1
lb_5[0] = 0
#ub = ub_5
#lb = lb_5

kl = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 ,0]
ku = [20, 24, 100, 20, 20, 20, 30, 48, 64, 100, 1, 1, 1, 1]
#ub = ku
#lb = kl


def SBCcross(p1, p2, l_b, u_b, distributionIndex=20, EPS=1.0e-14):
    c1 = p1
    c2 = p2
    if np.random.rand() <= 0.9:
        if abs(p1-p2) > EPS:
            if p1 < p2:
                y1 = p1
                y2 = p2
            else:
                y1 = p2
                y2 = p1

            lowerBound = l_b
            upperBound = u_b

            rand = np.random.rand()
            beta = 1.0 + (2.0 * (y1 - lowerBound) / (y2 - y1))
            alpha = 2.0 - math.pow(beta, -(distributionIndex + 1.0))

            if rand <= (1.0 / alpha):
                betaq = math.pow(rand * alpha, (1.0 / (distributionIndex + 1.0)))
            else:
                betaq = math.pow(1.0 / (2.0 - rand * alpha), 1.0 / (float(distributionIndex) + 1.0))
            
            c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))

            beta = 1.0 + (2.0 * (upperBound - y2) / (y2 - y1))
            alpha = 2.0 - math.pow(beta, -(distributionIndex + 1.0))

            if rand <= (1.0 / alpha):
                betaq = math.pow((rand * alpha), (1.0 / (distributionIndex + 1.0)))
            else:
                betaq = math.pow(1.0 / (2.0 - rand * alpha), 1.0 / (float(distributionIndex) + 1.0))
            
            c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

    return  c1, c2
 
    
def PolynomialMutation(p, l_b, u_b, distributionIndex=20):
    for i in range(len(p)):
        yl = l_b[i]
        yu = u_b[i]  
        y = p[i]
        if np.random.rand() <= 1 / len(p):
            if yl == yu:
                y = yl
            else:
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                rnd = np.random.rand()
                mutPow = 1.0 / (distributionIndex + 1.0);
                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (math.pow(xy, distributionIndex + 1.0));
                    deltaq = math.pow(val, mutPow) - 1.0;
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (math.pow(xy, distributionIndex + 1.0))
                    deltaq = 1.0 - math.pow(val, mutPow)
    
                p[i] = y + deltaq * (yu - yl)
        
    return p
        
        
def ga_deal(popu):
    for i in range(int(popu.shape[0]/2)):
        for j in range(int(popu.shape[1])):
            popu[2*i][j], popu[2*i+1][j] = SBCcross(popu[2*i][j], popu[2*i+1][j], l_b=lb[j], u_b=ub[j])
    for i in range(popu.shape[0]):
        popu[i] = PolynomialMutation(popu[i], l_b=lb, u_b=ub)
    
    return popu


def function(popu):
    p = popu
    while popu.shape[0] < 100:
        popu = np.vstack((popu, ga_deal(p)))
        
    return popu
