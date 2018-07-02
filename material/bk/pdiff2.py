# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:27:06 2018

@author: Administrator
"""

import nlopt
#from numpy import *
import numpy as np
import scipy
#########################tools
def read_xmu():
    xmudata=np.loadtxt('./feffrun/xmu.dat',comments='#')
    energy=xmudata[:,0]
    xmu=xmudata[:,3]                 
    return (energy,xmu)
def read_exp():
    data=np.loadtxt('./inp/gs.exp',comments='#')
    energy=data[:,0]
    exp=data[:,1]
    return (energy,exp)
def rfac(x1,y1,x2,y2):
    max1=max(x1)
    max2=max(x2)
    min1=min(x1)
    min2=min(x2)
    a=max(min1,min2)
    b=min(max1,max2)
    nip=100
    x=np.linspace(a,b,nip)
    f1=scipy.interpolate.interp1d(x1,y1,kind="slinear") 
    f2=scipy.interpolate.interp1d(x2,y2,kind="slinear")
    y1p=f1(x)
    y2p=f2(x)
    diff=y1p-y2p
    diff=pow(diff,2)
    result=sum(diff)
    result=np.sqrt(result)/nip
    return result
#########################crystal related
def R_3cR(x0,y0,z0):
    #R-3cR space group operation
    cor=np.array([[x0,y0,z0]]);#注意直接建立二维的矩阵coordination
    x=x0;y=y0;z=z0;
###
    current=[z,x,y]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
    current=[y,z,x]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
    current=[-y+1/2,-x+1/2,-z+1/2]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
    current=[-x+1/2,-z+1/2,-y+1/2]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
    current=[-z+1/2,-y+1/2,-x+1/2]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
    current=[-x,-y,-z]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
    current=[-z,-x,-y]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
    current=[-y,-z,-x]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
    current=[y+1/2,x+1/2,z+1/2]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
    current=[x+1/2,z+1/2,y+1/2]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
    current=[z+1/2,y+1/2,x+1/2]
    if(ifrepeat(cor,current)==0):
        cor=np.row_stack((cor,current))
###
    unit=cor
    unit=np.mat(unit)
    return unit
#
def period(all,unit,v,num):
    for i in range(-num,num+1):
        for j in range(-num,num+1):
            for k in range(-num,num+1):
                idx=[i,j,k]
                if not(i==0 and j==0 and k==0):#去掉 unit 本身
                    
                    extend=ijk(unit,v,idx)
                    all = fil(all,extend)
                    #all=np.vstack((all,extend))
    return all
def fil(all,extend):
    idx=extend.shape
    n1e=idx[0]
    #[n1e,n2e]=size(extend);
    for i in range(n1e):
         if(ifrepeat_mat(all,extend[i,:])==0):
             all=np.vstack((all,extend[i,:]))
    return all

def ijk(unix,v,idx):
    v1=v[0,:]
    v2=v[1,:]
    v3=v[2,:]
    extend=unit.copy()#深拷贝
    i=idx[0]
    j=idx[1]
    k=idx[2]
    mn=unit.shape
    for ii in range(mn[0]):
        extend[ii,:]=unit[ii,:]+i*v1+j*v2+k*v3
    return extend

def ifrepeat(mat,row):
    idx=mat.shape#mat should be numpy array
    n1=idx[0]
    result=0;
    z=1e-4;#tolerance
    for i in range(n1):
        if(abs(mat[i,0]-row[0])<z and abs(mat[i,1]-row[1])<z and abs(mat[i,2]-row[2])<z):
            result=1;
    #end for
    return result

def ifrepeat_mat(mat,row):
    #when row is (1,3 dim)
    #遍历mat,判断row是否与mat的某一行一致 tol为z
    idx=mat.shape#mat should be numpy array
    n1=idx[0]
    result=0;
    z=1e-4;#tolerance
    for i in range(n1):
        if(abs(mat[i,0]-row[0,0])<z and abs(mat[i,1]-row[0,1])<z and abs(mat[i,2]-row[0,2])<z):
            result=1;
    #end for
    return result

def bsxfun_m(all,ori):
    idx=all.shape
    n=idx[0]
    all=np.array(all)#一定要转换不转换不对
    ori=np.array(ori)
    for i in range(n):
        all[i,:]=all[i,:]-ori
    return all
    
#########################    
def myobj(x,grad):
    es=x[0]
    ratio=x[1]
    dxmu=read_xmu()
    dexp=read_exp()
    result=rfac(dmu[0]+es,dmu[1]*ratio,dexp[0],dexp[1])
    return result





#########################NLOPT example
def myfunc(x, grad):
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 0.5 / sqrt(x[1])
    return sqrt(x[1])
def myconstraint(x, grad, a, b):
    if grad.size > 0:
        grad[0] = 3 * a * (a*x[0] + b)**2
        grad[1] = -1.0
    return (a*x[0] + b)**3 - x[1]
##########################
dmu=read_xmu()
dexp=read_exp()
aa=rfac(dmu[0],dmu[1],dexp[0],dexp[1])

xf=0.1
yf=0.1
zf=0.1
xo=-0.3
yo=0.8
zo=0.25
###
alpha=55.23/180*np.pi
beta=55.23/180*np.pi
gamma=55.23/180*np.pi
a=5.431
b=5.431
c=5.431
###
num=3 #number of the peroid
###
V=a*b*c*np.sqrt(1-pow(np.cos(alpha),2)-pow(np.cos(beta),2)-pow(np.cos(gamma),2)+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
tr1=[a,b*np.cos(gamma),c*np.cos(beta)]
tr1=np.array(tr1)
tr2=[0,b*np.sin(gamma),c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)]
tr2=np.array(tr2)
tr3=[0,0,V/(a*b*np.sin(gamma))]
tr3=np.array(tr3)
tra=np.stack((tr1,tr2,tr3))
tra=np.mat(tra)#mat
v1=a*np.array([1,0,0])
v2=b*np.array([np.cos(gamma),np.sin(gamma),0])
v3=[np.cos(beta),(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),np.sqrt(1+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)-pow(np.cos(alpha),2)-pow(np.cos(beta),2)-pow(np.cos(gamma),2))/np.sin(gamma)]
v3=c*np.array(v3)
vv=np.stack((v1,v2,v3))
v=vv
v=np.mat(v)#mat
##########################
atom1=R_3cR(xf,yf,zf)
atom2=R_3cR(xo,yo,zo)

unit=atom1.T
unit=tra*unit
unit=unit.T#分数坐标转换直角坐标 每行为一条数据
all_atom1=unit#extend=ijk(unit,v,[1,1,1])
all_atom1=period(all_atom1,unit,v,num)#直角坐标平移

unit=atom2.T
unit=tra*unit
unit=unit.T#分数坐标转换直角坐标 每行为一条数据
all_atom2=unit
all_atom2=period(all_atom2,unit,v,num)#直角坐标平移
################ori
ori=all_atom1[0,:]
all_atom1=bsxfun_m(all_atom1,ori)
all_atom2=bsxfun_m(all_atom2,ori)
pot1=26*np.ones(all_atom1.shape[0])
all_atom1=np.column_stack((all_atom1,pot1.T))
pot2=8*np.ones(all_atom2.shape[0])
all_atom2=np.column_stack((all_atom2,pot2.T))
all=np.row_stack((all_atom1,all_atom2))

################dis
output=all.copy()
dis=np.array([])
for i in range(all.shape[0]):
    xa=all[i,0]
    ya=all[i,1]
    za=all[i,2]
    disa=np.sqrt(pow(xa,2)+pow(ya,2)+pow(za,2))
    dis=np.append(dis,disa)
output=np.column_stack((output,dis.T))
#sort 
output=np.array(output)
output=output[output[:,4].argsort()]
#
f1 = open("out1.txt", "w")
ml=output.shape[0]
for ll in range(ml):
    print("%0.3f %0.3f %0.3f %d %0.3f" % (output[ll,0],output[ll,1],output[ll,2],output[ll,3],output[ll,4]), file = f1)
f1.close()

















#opt = nlopt.opt(nlopt.LD_MMA, 2)
#opt.set_lower_bounds([-float('inf'), 0])
#opt.set_min_objective(myfunc)
#opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,2,0), 1e-8)
#opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,-1,1), 1e-8)
#opt.set_xtol_rel(1e-4)
#x = opt.optimize([1.234, 5.678])
#minf = opt.last_optimum_value()
#print("optimum at ", x[0], x[1])
#print("minimum value = ", minf)
#print("result code = ", opt.last_optimize_result())
###########################################################
#opt = nlopt.opt(nlopt.GN_DIRECT_L, 2)
#opt.set_lower_bounds([-5,0.5])
#opt.set_upper_bounds([5,2])
#opt.set_min_objective(myobj)
#opt.set_xtol_rel(1e-4)
#opt.set_maxeval(100)
#x = opt.optimize([0, 1])
#minf = opt.last_optimum_value()
#print("optimum at ", x[0], x[1])
#print("minimum value = ", minf)
#print("result code = ", opt.last_optimize_result())

