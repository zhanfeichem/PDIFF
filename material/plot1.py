# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:20:59 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

dat=np.loadtxt("./obj/ISRES1_o_obj.txt")

dat=np.loadtxt("./obj/o_obj.txt")
x=dat[:,0]
y=dat[:,3]
plt.plot(x,y,'o')
plt.ylim(0,0.02)
