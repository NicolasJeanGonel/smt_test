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


# # f1

# # with mixed variables

# # In[9]:


# N_obs = 10
# Nmax = 1024
# f = fcn1
# dx = np.sqrt(2)/3.0
# xt = np.linspace(xMin + dx, xMax - dx , N_obs)
# print(xt)
# yt = f(xt)
# design_space = DesignSpace(
#     [
#         FloatVariable(xMin, xMax ),
#         #CategoricalVariable(xt),
#         #OrdinalVariable(xt),
#     ]
# )
# sm = MixedIntegerKrigingModel(
#     surrogate=KRG(design_space=design_space, theta0=[1e-2], hyper_opt="Cobyla")
# )
# sm.set_training_values(xt,yt)
# sm.train()

# num = 1024
# x = np.linspace(xMin,xMax,num)
# y = sm.predict_values(x)
# v = sm.predict_variances(x)
# sigma = np.sqrt(v)

# print(xt)
# plt.figure()
# plt.plot(x, f(x), 'r:', label=r'$f(x)$')
# plt.plot(xt, yt, 'r.', markersize=10, label='Observations')
# plt.plot(x, y, 'b-', label='Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
# np.concatenate([y - 1.9600 * sigma,
# (y + 1.9600 * sigma)[::-1]]),
# alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-15, 20)
# plt.legend(loc='upper left')
# plt.show()


# f3

# with mixed variables

# In[10]:


# N_obs = 45
# Nmax = 1024
# f = fcn3
# dx = np.sqrt(2)/3.0
# xt = np.linspace(xMin + dx, xMax - dx , N_obs)
# yt = f(xt)
# design_space = DesignSpace(
#     [
#         FloatVariable(xMin, xMax),
#         #CategoricalVariable(xt),
#     ]
# )
# sm = MixedIntegerKrigingModel(
#     surrogate=KRG(design_space=design_space, theta0=[1e-2], hyper_opt="Cobyla")
# )
# sm.set_training_values(xt,yt)
# sm.train()

# num = 1024
# x = np.linspace(xMin,xMax,num)
# y = sm.predict_values(x)
# v = sm.predict_variances(x)
# sigma = np.sqrt(v)


# plt.figure()
# plt.plot(x, f(x), 'r:', label=r'$f(x)$')
# plt.plot(xt, yt, 'r.', markersize=10, label='Observations')
# plt.plot(x, y, 'b-', label='Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
# np.concatenate([y - 1.9600 * sigma,
# (y + 1.9600 * sigma)[::-1]]),
# alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-15, 20)
# plt.legend(loc='upper left')
# plt.show()


# ### modif proposed


# from smt.sampling_methods import LHS
# from smt.applications.mixed_integer import MixedIntegerSamplingMethod
# N_obs = 10
# design_space = DesignSpace(
#     [
#         #FloatVariable(xMin, xMax),
#         IntegerVariable(xMin, xMax),
#         #CategoricalVariable(xt),
#     ]
# )
# sampling = MixedIntegerSamplingMethod(LHS,design_space, criterion="ese", random_state=42)

# xdoe = sampling(N_obs)
# f = fcn3
# ydoe = f(xdoe)

# sm = MixedIntegerKrigingModel(
#     surrogate=KRG(design_space=design_space, theta0=[1e-2], hyper_opt="Cobyla")
# )
# sm.set_training_values(xdoe,ydoe)
# sm.train()


# num = 1024
# x = np.linspace(xMin,xMax,num)
# y = sm.predict_values(x)
# v = sm.predict_variances(x)
# sigma = np.sqrt(v)


# plt.figure()
# plt.plot(x, f(x), 'r:', label=r'$f(x)$')
# plt.plot(xdoe, ydoe, 'g.', markersize=10, label='Observations')
# plt.plot(x, y, 'b-', label='Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
# np.concatenate([y - 1.9600 * sigma,
# (y + 1.9600 * sigma)[::-1]]),
# alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-15, 20)
# plt.legend(loc='upper left')
# plt.show()

# f5

# Exponential Squared Sine

# In[11]:
eps=1e-8
x_der=48
N_obs = 40
Nmax = 1024
f = fcn5
dx = np.sqrt(2)/3.0
xt = np.linspace(xMin + dx, xMax - dx , N_obs)
yt = f(xt)
#noise0 = [1.0], eval_noise = True, ;
sm = KRG(poly = 'linear',corr='squar_sin_exp',hyper_opt = 'TNC', xlimits = np.array([[xMin ,xMax ]]), noise0 = [0.0001], eval_noise = True, use_het_noise = True, n_start = 1000, theta0 = [1.0,5.0])
sm.set_training_values(xt,yt)
sm.train()

num = 1024
x = np.linspace(1/2*xMin,2*xMax,num)
y = sm.predict_values(x)
dy=sm.predict_derivatives(x,0)
v = sm.predict_variances(x)
v_diff=sm.predict_variances(x+eps)
dv=sm.predict_variance_derivatives(x,0)

sigma = np.sqrt(v)
y_diff=sm.predict_values(x+eps)
def tangente(x,x_0,y,alpha):
    return alpha*(x-x_0)+y
# for i in range(len(dy)):
#     if dy[i]-(y_diff[i]-y[i])/eps>1e-3:
#         print("indice d'échec=",i)
# print(np.allclose(dy,(y_diff-y)/eps,rtol=1e-3))
# print("d y_diff=",(y_diff[x_der]-y[x_der])/eps,"\ndy=",dy[x_der])
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x)$')
plt.plot(x,tangente(x,x[x_der],y[x_der],dy[x_der]),label="tangente pour dérivé")
plt.plot(x,tangente(x,x[x_der],y[x_der],(y_diff[x_der]-y[x_der])/eps),label="tangente pour différence finie")
plt.axvline(x[x_der])
plt.plot(xt, yt, 'r.', markersize=10, label='Observations')
plt.plot(x, y, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
np.concatenate([y - 1.9600 * sigma,
(y + 1.9600 * sigma)[::-1]]),
alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()
def k(theta1,theta2,d):
    return np.exp(-theta1*np.sin(theta2*d)**2)
def dk(theta1,theta2,d):
    return -theta1*theta2*np.sin(2*theta2*d)*k(theta1,theta2,d)
plt.figure()
plt.plot(x,k(1,13,x))
plt.axvline(x[x_der])
plt.plot(x,tangente(x,x[x_der],k(1,13,x[x_der]),dk(1,13,x[x_der])))
plt.show()
plt.figure()
plt.plot(x,v)
plt.plot(x,tangente(x,x[x_der],v[x_der],dv[x_der]),label="tangente pour dérivé")
plt.plot(x,tangente(x,x[x_der],v[x_der],dv[x_der]-(v_diff[x_der]-v[x_der])/eps),label="tangente pour différence fini")
plt.axvline(x[x_der])
plt.legend(loc='upper left')
plt.show()
print(dv)
print((v_diff-v)/eps)
for i in range(len(dv)):
    if (dv[i]-(v_diff[i]-v[i])/eps)/dv[i]>1e-1:
        print("indice d'échec=",i)
        print("v_i=",v[i])
        print("dv[i]=",dv[i],"dv_diff=",(v_diff[i]-v[i])/eps)
print(np.allclose(dv,(v_diff-v)/eps,rtol=1e-1))
# In[ ]:





# In[ ]:




