# -*- coding: utf-8 -*-
import numpy as np
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
#B:total budget
#P:first price
#P_2:next price
#R:return rate
#Rcov:covariance of R
#N:number of assets
#T:number of time
#theta:weight
#->1:return 2:risk aversion 3:constrain

#setting parameters
T = 6
N = 10
B = 150
theta1 = 0.3
theta2 = 0.3
theta3 = 3.0

np.random.seed(5)

#make random prices
P = np.random.randint(1,100,(T,N))

#calculate return rate
R = np.zeros((T-1,N))
for j in range(N):
    for i in range(T-1):
        R[i,j] = (P[i+1,j] -P[i,j]) / P[i,j]

#calculate covariance of R        
#axis = 0:calculate over column,bias = 1:normalization by N
Ravg = np.average(R, axis=0)
Rcov = np.cov(R, rowvar=0, bias=1)

#make QUBO
Qmirror = np.empty((N,N))
Q = {}
for i in range(N):
    for j in range(N):
        if i == j:
            Qmirror[i,j] =  ( theta2*Rcov[i,j] + theta3*P[-1,i]*P[-1,i] - theta1*Ravg[i] -2*theta3*B*P[-1,i] ) / 2
            Q[i,j] = Qmirror[i,j]
        else:
            Qmirror[i,j] =  ( theta2*Rcov[i,j] + theta3*P[-1,i]*P[-1,j] ) / 4
            Q[i,j] = Qmirror[i,j]

#sampling with D-wave machine           
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q, num_reads=100)
for sample in response.samples():
    print(sample)

#check sum
sum = 0
for i in range(N):
    if response.record['sample'][0,i] == 1:
        sum += P[-1,i]

print(response.record['sample'][0])        
print(sum)