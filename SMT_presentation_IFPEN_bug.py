#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# In[2]:


xMin = 0
xMax = 10
def fcn1(x): #smooth function
    '''The function to predict.'''
    return 2.0 * x * np.sin( x )

def fcn2(x): #local non-derivative function
    '''The function to predict.'''
    return fcn_interp(x)

def fcn3(x): #local discontinous function
    '''The function to predict.'''
    return np.floor(fcn1(x)*0.2)*5
x_c = np.linspace(xMin - 5,xMax + 5,10)
fcn_interp = interp1d(x_c, fcn1(x_c), kind='linear')

def fcn4(x): #smooth function + linear
    '''The function to predict.'''
    return  np.exp(-(x-5.0)**2)*x * np.sin(x) + 1.5*x
def fcn5(x): #linear + local periodic function
    '''The function to predict.'''
    return  np.sin(10*x)# + 1.5*x
def fcn6(x): #linear + local periodic function + local discontinous function
    '''The function to predict.'''
    return 1.5*np.floor(x / 2.0 )* 2.0  + np.sin(x * 10)


# In[3]:


x = np.linspace(xMin,xMax,512)
plt.figure()
plt.plot(x,fcn1(x),label = 'f1')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()


# In[4]:


x = np.linspace(xMin,xMax,512)
plt.figure()
plt.plot(x,fcn2(x),label = 'f2')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()


# In[5]:


x = np.linspace(xMin,xMax,512)
plt.figure()
plt.plot(x,fcn3(x),label = 'f3')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()


# In[6]:


x = np.linspace(xMin,xMax,512)
plt.figure()
plt.plot(x,fcn4(x),label = 'f4')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()


# In[7]:


x = np.linspace(xMin,xMax,512)
plt.figure()
plt.plot(x,fcn5(x),label = 'f5')
plt.plot(x,fcn6(x),label = 'f6')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()


# In[8]:


from smt.surrogate_models import RBF
from smt.surrogate_models import IDW
from smt.surrogate_models import RMTB
from smt.surrogate_models import RMTC
from smt.surrogate_models import KRG
from smt.surrogate_models import KRG
from smt.applications.mixed_integer import MixedIntegerKrigingModel
from smt.utils.design_space import DesignSpace, FloatVariable, IntegerVariable, OrdinalVariable, CategoricalVariable
from smt.surrogate_models import GENN
from smt.sampling_methods import LHS


# f5

# Exponential Squared Sine

# In[11]:
# eps=1e-8
# x_der=48
# N_obs = 40
# Nmax = 1024
# f = fcn5
# dx = np.sqrt(2)/3.0
# xt = np.linspace(xMin + dx, xMax - dx , N_obs)
# yt = f(xt)
# #noise0 = [1.0], eval_noise = True, ;
# sm = KRG(poly = 'linear',corr='squar_sin_exp',hyper_opt = 'TNC', xlimits = np.array([[xMin ,xMax ]]), noise0 = [0.0001], eval_noise = True, use_het_noise = True, n_start = 1000, theta0 = [1.0,5.0])
# sm.set_training_values(xt,yt)
# sm.train()

# num = 1024
# x = np.linspace(1/2*xMin,2*xMax,num)
# y = sm.predict_values(x)
# dy=sm.predict_derivatives(x,0)
# v = sm.predict_variances(x)
# v_diff=sm.predict_variances(x+eps)
# dv=sm.predict_variance_derivatives(x,0)

# sigma = np.sqrt(v)
# y_diff=sm.predict_values(x+eps)
# def tangente(x,x_0,y,alpha):
#     return alpha*(x-x_0)+y
# # for i in range(len(dy)):
# #     if dy[i]-(y_diff[i]-y[i])/eps>1e-3:
# #         print("indice d'échec=",i)
# # print(np.allclose(dy,(y_diff-y)/eps,rtol=1e-3))
# # print("d y_diff=",(y_diff[x_der]-y[x_der])/eps,"\ndy=",dy[x_der])
# plt.figure()
# plt.plot(x, f(x), 'r:', label=r'$f(x)$')
# plt.plot(x,tangente(x,x[x_der],y[x_der],dy[x_der]),label="tangente pour dérivé")
# plt.plot(x,tangente(x,x[x_der],y[x_der],(y_diff[x_der]-y[x_der])/eps),label="tangente pour différence finie")
# plt.axvline(x[x_der])
# plt.plot(xt, yt, 'r.', markersize=10, label='Observations')
# plt.plot(x, y, 'b-', label='Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
# np.concatenate([y - 1.9600 * sigma,
# (y + 1.9600 * sigma)[::-1]]),
# alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
# plt.legend(loc='upper left')
# plt.show()
# def k(theta1,theta2,d):
#     return np.exp(-theta1*np.sin(theta2*d)**2)
# def dk(theta1,theta2,d):
#     return -theta1*theta2*np.sin(2*theta2*d)*k(theta1,theta2,d)
# plt.figure()
# plt.plot(x,k(1,13,x))
# plt.axvline(x[x_der])
# plt.plot(x,tangente(x,x[x_der],k(1,13,x[x_der]),dk(1,13,x[x_der])))
# plt.show()
# plt.figure()
# plt.plot(x,v)
# plt.plot(x,tangente(x,x[x_der],v[x_der],dv[x_der]),label="tangente pour dérivé")
# plt.plot(x,tangente(x,x[x_der],v[x_der],dv[x_der]-(v_diff[x_der]-v[x_der])/eps),label="tangente pour différence fini")
# plt.axvline(x[x_der])
# plt.legend(loc='upper left')
# plt.show()
# print(dv)
# print((v_diff-v)/eps)
# for i in range(len(dv)):
#     if (dv[i]-(v_diff[i]-v[i])/eps)/dv[i]>1e-1:
#         print("indice d'échec=",i)
#         print("v_i=",v[i])
#         print("dv[i]=",dv[i],"dv_diff=",(v_diff[i]-v[i])/eps)
# print(np.allclose(dv,(v_diff-v)/eps,rtol=1e-1))
# In[ ]:
from matplotlib import cm

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

def fcn7(X,Y):
    # sin + linear trend
    y = (2*np.sin(X))+np.sin(Y) + 2*Y
          # + linear trend
    return y
def pb(x):
    # sin + linear trend
    y = (
        np.atleast_2d(np.sin(x[:, 0])).T
        + np.atleast_2d(2 * x[:, 0] + 5 * x[:, 1]).T
        + 10
    )  # + linear trend
    return y

# xlimits = np.array([[-5, 10], [-5, 10]])
# sampling = LHS(xlimits=xlimits, random_state=42)
# xt = sampling(12)
# print(xt)
# yt = fcn7(xt)

# X=xt[:,0]
# Y=xt[:,1]
# Z=yt

# X1 = np.arange(-10, 10, 0.1)
# Y1 = np.arange(-10, 10, 0.1)
# X, Y = np.meshgrid(X1, Y1)
# Z = fcn7(X,Y)
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()


sm = KRG(poly = 'linear',corr='squar_sin_exp',  noise0 = [0.0001], eval_noise = True, use_het_noise = True, n_start = 1000)
# In[ ]:
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
xlimits = np.array([[-5, 10], [-5, 10]])
sampling = LHS(xlimits=xlimits, random_state=42)
xt = sampling(12)
yt=pb(np.array([xt[:,0],xt[:,1]]).T)
print(xt,yt)
sm.set_training_values(xt,yt)
sm.train()
# print(xt)
x_test=np.linspace(-5,10,50)
pred=[]
dpredx=[]
dpredy=[]

for x0 in x_test:
    for x1 in x_test:
        pred.append(sm.predict_values(np.array([[x0,x1]])))
        dpredy.append(sm.predict_derivatives(np.array([[x0,x1]]),1))
        dpredx.append(sm.predict_derivatives(np.array([[x0,x1]]),0))
        
pred=np.array(pred)
dpredy=np.array(dpredy)
dpredx=np.array(dpredx)
pred=pred.reshape((50,50)).T
dpredy=dpredy.reshape((50,50)).T

dpredx=dpredx.reshape((50,50)).T
def plan(x,y,a,b,alpha1,alpha2,z):
    return z+alpha1*(x-a)+alpha2*(y-b)
X,Y = np.meshgrid(x_test,x_test)
z_tan=plan(X,Y,x_test[10],x_test[10],dpredx[10,10],dpredy[10,10],pred[10,10])
fig = plt.figure(figsize=(15, 10))
ax =  fig.add_subplot(projection='3d')
ax.scatter(x_test[10],x_test[10],pred[10,10],zdir='z', c='b')
# ax.scatter(xt[:,0], xt[:,1], yt, zdir='z', marker='x', c='b', s=200, label='DOE')
surf = ax.plot_surface(X, Y, pred, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,alpha=0.5)
surf = ax.plot_surface(X, Y, z_tan, cmap=cm.Spectral_r,
                       linewidth=0, antialiased=False,alpha=0.5)
# x_test=np.array([np.arange(-5,10,1),np.arange(-5,10,1)]).T
# print(x_test)
# y = sm.predict_values(x_test).reshape(len(x_test))
# print(yt)
# print(y)
# print(y.shape,x_test.shape)
# surf = ax.plot_trisurf(xt[:,0],xt[:,1],yt, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_trisurf(x_test[:,0],x_test[:,1],y, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()


