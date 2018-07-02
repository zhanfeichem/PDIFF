# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:27:06 2018

@author: Fei Zhan 
2018-4-25 (1)all using ndarray instead of matrix object (2) give up repeat_mat function due to matrix
"""
from __future__ import print_function
import os
import nlopt

import numpy as np
import scipy
import scipy.optimize
import scipy.interpolate
####global variable
#dxmu=np.array([])
#dexp=np.array([])

RUNFEFF= True
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
    unit=np.array(unit)
    #unit=np.mat(unit)
    return unit
#
def period(all,unit,v,num):
    all=np.array(all)
    unit=np.array(unit)
    v=np.array(v)
    #
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
    all=np.array(all)
    extend=np.array(extend)
    #
    idx=extend.shape
    n1e=idx[0]
    #[n1e,n2e]=size(extend);
    for i in range(n1e):
         #if(ifrepeat_mat(all,extend[i,:])==0):
         if(ifrepeat(all,extend[i,:])==0):
             all=np.vstack((all,extend[i,:]))
             
    return np.array(all)

def ijk(unit,v,idx):
    unit=np.array(unit)
    v=np.array(v)
    idx=np.array(idx)
    #
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
    return np.array(extend)

def ifrepeat(mat,row):
    #retirn value 0 or 1
    mat=np.array(mat)
    row=np.array(row)
    #
    idx=mat.shape#mat should be numpy array
    n1=idx[0]
    result=0;
    z=1e-4;#tolerance
    for i in range(n1):
        if(abs(mat[i,0]-row[0])<z and abs(mat[i,1]-row[1])<z and abs(mat[i,2]-row[2])<z):
            result=1;
    #end for
    return result

#def ifrepeat_mat(mat,row):
#    #when row is (1,3 dim)
#    #遍历mat,判断row是否与mat的某一行一致 tol为z
#    mat=np.array(mat)
#    row=np.array(row)
#    #
#    idx=mat.shape#mat should be numpy array
#    n1=idx[0]
#    result=0;
#    z=1e-4;#tolerance
#    for i in range(n1):
#        if(abs(mat[i,0]-row[0,0])<z and abs(mat[i,1]-row[0,1])<z and abs(mat[i,2]-row[0,2])<z):
#            result=1;
#    #end for
#    return result

def bsxfun_m(all,ori):
    all=np.array(all)
    ori=np.array(ori)
    #
    idx=all.shape
    n=idx[0]
    all=np.array(all)#一定要转换不转换不对
    ori=np.array(ori)
    for i in range(n):
        all[i,:]=all[i,:]-ori
    return np.array(all)
    
#########input and output related function################ 
    
##########################################################

    
def en_obj(x):
    #def en_obj(x,grad,fdat=(dxmu,dexp)):
    #energy shift and normalization obj
    es=x[0]
    ratio=x[1]
#    dxmu=fdat[0]
#    dexp=fdat[1]
    dxmu=read_xmu()
    dexp=read_exp()
    result=rfac(dxmu[0]+es,dxmu[1]*ratio,dexp[0],dexp[1])
    return result

def en_obj_nlopt(x,grad):
    return en_obj(x)

def rfac_opt():
    ###NLOPT
#    opt = nlopt.opt(nlopt.GN_DIRECT_L, 2)
#    opt.set_min_objective(en_obj)#obj function
#    opt.set_lower_bounds([-5,0.5])
#    opt.set_upper_bounds([5,2])
#    opt.set_xtol_rel(1e-4)
#    opt.set_maxeval(100)
#    x = opt.optimize([0, 1])#initial guess
#    minf = opt.last_optimum_value()
    ###SCIPY
    bnds = ((-5, 5), (0.8,2.0))
    x0=(0,1.0)
    res = scipy.optimize.minimize(en_obj, x0, method='SLSQP',bounds=bnds,options={'maxiter':100,})
    minf=res.fun
    #####################################
    return minf
    





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
#dmu=read_xmu()
#dexp=read_exp()
#aa=rfac(dmu[0],dmu[1],dexp[0],dexp[1])
#####set value
###feff potemtial

    #END  myobj_nlopt
def myobj(x):
#def myobj(x,grad):
    pot1_val=1
    pot2_val=2
    nfeff=500
    ###fit the abc
    
    a=x[0]
    b=x[0]
    c=x[0]
    ###########################
#    a=5.431
#    b=5.431
#    c=5.431
    alpha=55.23/180*np.pi
    beta=55.23/180*np.pi
    gamma=55.23/180*np.pi
    ###
    xf=0.1
    yf=0.1
    zf=0.1
    xo=-0.3
    yo=0.8
    zo=0.25
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
    v=np.array(vv)
    #v=np.mat(v)#mat
    ##########################
    atom1=R_3cR(xf,yf,zf)
    atom2=R_3cR(xo,yo,zo)
    
    unit=atom1.T
    #unit=tra*unit
    unit=np.dot(tra,unit)
    unit=unit.T#分数坐标转换直角坐标 每行为一条数据
    all_atom1=unit#extend=ijk(unit,v,[1,1,1])
    all_atom1=period(all_atom1,unit,v,num)#直角坐标平移
    
    unit=atom2.T
    #unit=tra*unit
    unit=np.dot(tra,unit)
    unit=unit.T#分数坐标转换直角坐标 每行为一条数据
    all_atom2=unit
    all_atom2=period(all_atom2,unit,v,num)#直角坐标平移
    ################ori
    ori=all_atom1[0,:]
    all_atom1=bsxfun_m(all_atom1,ori)
    all_atom2=bsxfun_m(all_atom2,ori)
    pot1=pot1_val*np.ones(all_atom1.shape[0])
    all_atom1=np.column_stack((all_atom1,pot1.T))
    pot2=pot2_val*np.ones(all_atom2.shape[0])
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
    
    f_minp = open("./inp/feffmodel.inp", "r") #model inp
    l_minp= f_minp.readlines() 
    f_minp.close()
    
    f_inp=open("./feffrun/feff.inp","w") # feff.inp in feffrun
    for line in l_minp:                          #依次读取每行  
        line = line.strip()                             #去掉每行头尾空白  
        print("%s" % (line),file=f_inp)
    for ll in range(nfeff):# jump the absorption center
        if (ll==0):
            print("0.0 0.0 0.0 0 ",file=f_inp)
        if (ll!=0):
            print("%0.3f %0.3f %0.3f %d " % (output[ll,0],output[ll,1],output[ll,2],output[ll,3]), file = f_inp)
    print("END",file=f_inp)
    f_inp.close()
    #saveinp
    os.chdir("./feffrun")
    if RUNFEFF:
        os.system("feffwin")
    os.chdir("..")
    
    
    xmudata=[]
    try:
        xmudata=np.loadtxt('./feffrun/xmu.dat',comments='#')#更改判断条件2018-4-29
    except:
        pass

    if len(xmudata):
    #if any(xmudata):
    #if os.path.exists("./feffrun/xmu.dat"):
        #SyntaxError: name 'dxmu' is used prior to global declaration
    #    global dxmu 
    #    global dexp
#        dxmu=read_xmu()
#        dexp=read_exp()
        obj=rfac_opt()
        #obj-opt
    else:
        obj=1001.0#主义必须给浮点数，NLOPT不自动转换
    #save par-obj 
    f_obj = open("o_obj.txt", "a")
    print("%0.3f %0.3f %0.3f %0.4f" % (a,b,c,obj), file = f_obj)
    f_obj.close()
    #clean inp dat bin files
    if os.path.exists("./feffrun/fms.bin"):
        os.remove("./feffrun/xmu.dat")
        os.remove("./feffrun/feff.inp")
        os.remove("./feffrun/apot.bin")
        os.remove("./feffrun/feff.bin")
        os.remove("./feffrun/fms.bin")
        os.remove("./feffrun/gg.bin")
        os.remove("./feffrun/phase.bin")
        os.remove("./feffrun/pot.bin")
    print("obj is",obj)
    return obj #ValueError: nlopt invalid argument

def myobj_nlopt(x,grad):
    return myobj(x)
#END of myobj function########################
#####run obj function
#obj=myobj_nlopt([3.5],[])
###Scipy
#bounds = [(4.0,7.5)]
#result = scipy.optimize.differential_evolution(func=myobj,bounds=bounds,maxiter=100)
#result = scipy.optimize.basinhopping(func=myobj,bounds=bounds,maxiter=2)
#####NLOPT
opt = nlopt.opt(nlopt.GN_ESCH,1)
opt.set_min_objective(myobj_nlopt)#obj function
#Fe2O3 a=b=c=5.431
opt.set_lower_bounds([3.0])
opt.set_upper_bounds([10.5])
opt.set_xtol_rel(1e-4)
opt.set_stopval(1e-4)
opt.set_maxeval(1000)
x = opt.optimize([3.0])
minf = opt.last_optimum_value()
print("optimum at ", x[0])
print("minimum value = ", minf)
print("result code = ", opt.last_optimize_result())



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