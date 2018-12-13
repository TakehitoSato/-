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
#theta:weight
#->1:return 2:risk aversion 3:constrain

N = 10
B = 200
theta1 = 0.1
theta2 = 0.1
theta3 = 0.8

np.random.seed(5)
P = np.random.randint(1,100,(N))
#Pavg = np.average(P)
P_2 = np.random.randint(1,100,(N))
#P_2avg = np.average(P_2)

Rcov = np.empty((N,N))
R = np.empty(N)
#calculate expected return rate
for i in range(N):
    R[i] = (P_2[i] - P[i]) / P[i]
Ravg = np.average(R)

#calculate covariance(and variance)
for i in range(N):
    for j in range(N):
        Rcov[i,j] = ( (R[i] - Ravg)*(R[j] - Ravg) ) / N

#make QUBO
Qmirror = np.empty((N,N))
Q = {}
for i in range(N):
    for j in range(N):
        if i == j:
            Qmirror[i,j] =  ( theta2*Rcov[i,j] + theta3*P[i]*P[i] - theta1*R[i] -2*theta3*B*P[i] )
            Q[i,j] = Qmirror[i,j]
        else:
            Qmirror[i,j] =  ( theta2*Rcov[i,j] + theta3*P[i]*P[j] )
            Q[i,j] = Qmirror[i,j]

#sampling with D-wave machine           
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q, num_reads=1000)
for sample in response.samples():
    print(sample)


sum = 0
returnsum = 0
for i in range(N):
    if response.record['sample'][0,i] == 1:
        sum += P[i]
        returnsum += P[i] * R[i]

print(response.record['sample'][0])        
print(sum)
print(returnsum)