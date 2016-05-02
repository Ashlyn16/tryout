# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 23:46:58 2016

@author: SHUBHAM
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from ode1 import solver
solver()

v0=10
k1=0.5
k2=0.35
#desclare the model
def mymodel(y,t):
    dy0=v0-k1*y[0]
    dy1=k1*y[0]-k2*y[1]
    
    return [dy0,dy1]
    
time =np.linspace(0.0, 20.0, 100.0)
yinit=np.array([0.0,0.0])
y=odeint(mymodel,yinit,time)
plt.plot(time,y[:,0],time,y[:,1])

plt.xlabel('t')
plt.ylabel('y')
plt.show()
    