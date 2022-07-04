#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:08:59 2021

@author: yiye
"""
import os 
# GPU control
os.environ['GPU_use'] = 'False'
if os.environ['GPU_use'] == 'False':
    import numpy as xp
else:
    import cupy as xp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import heatmap
import time
from homotopy_algo import *
from utilities import *
import pickle 

with open('dt_MTS_04_26','rb') as fp:
    dt_MTS, ind = pickle.load(fp)
    
T = 127*12-1
N = len(ind)
F = 4
M = 12

dtype = np.float64

######## Hyperparameter control ########
lambda_0 = xp.float64(0.03).astype(dtype)
eta = xp.float64(1e-5).astype(dtype) # step size for lambda updating


# Preparation to start the online learning 
dim = int(N*F + .5*F*(F-1) + .5*N*(N-1)) # Dimension of space K
Gamma0 = np.zeros(shape = (dim, dim)) # The structured Gamma0_hat used in HomoAlgos
gamma1 = np.zeros(shape = (dim, 1)) # The structured Gamma1_hat used in HomoAlgos
Gamma0_hat, Gamma1_hat = np.zeros(shape = (N*F, N*F)), np.zeros(shape = (N*F, N*F))
Gamma0_b_hat, Gamma1_b_hat = np.zeros(shape = (N*F, N*F)), np.zeros(shape = (N*F, N*F))
p_m = np.zeros(12)  
x_lmean = np.zeros(shape = (N*F, M))
x_umean = np.zeros(shape = (N*F, M))
for t in np.arange(1, 20 + 1):
    x_old_obs = dt_MTS[:,:,int(t-1)].reshape(-1, 1, order = 'F').astype(dtype)
    x_new_obs = dt_MTS[:,:,int(t)].reshape(-1, 1, order = 'F').astype(dtype)
    
    m = t % M # current month
    p_m[m] = p_m[m] + 1 
    
    # estimation of monthly means
    x_umean[:,m] = (p_m[m]-1)*x_umean[:,m]/p_m[m] + x_new_obs.flatten()/p_m[m]
    x_lmean[:,m-1] = (p_m[m]-1)*x_lmean[:,m-1]/p_m[m] + x_old_obs.flatten()/p_m[m]

    Gamma0_b_hat = (t-1)*Gamma0_b_hat/t + x_old_obs.dot(x_old_obs.T)/t
    Gamma1_b_hat = (t-1)*Gamma1_b_hat/t + x_new_obs.dot(x_old_obs.T)/t 
    
    X_old_obs = node_mat(N, F, x_old_obs)
    Gamma0 = (t-1)*Gamma0/t + X_old_obs.dot(X_old_obs.T)/t
    gamma1 = (t-1)*gamma1/t + X_old_obs.dot(x_new_obs)/t
    
Gamma0_hat = Gamma0_b_hat - np.dot(x_lmean[:,np.arange(M)-1]*p_m, x_lmean[:,np.arange(M)-1].T)/t
Gamma1_hat = Gamma1_b_hat - np.dot(x_umean*p_m, x_lmean[:,np.arange(M)-1].T)/t
for m in range(M) :
    X_lmean = node_mat(N, F, x_lmean[:,m-1])
    Gamma0 = Gamma0 - p_m[m]*X_lmean.dot(X_lmean.T)/t
    gamma1 = gamma1 - p_m[m]*X_lmean.dot(x_umean[:,m].reshape(-1,1))/t
    
# Start Approach 2
D_hat, A_N_hat, A_F_hat = xp.zeros((N,F), dtype), xp.zeros((N,N), dtype), xp.zeros((F,F), dtype)
epsilon = xp.float64(5e-5).astype(dtype)
D_hat, A_N_hat, A_F_hat, n = PGD_gl(D_hat, A_N_hat, A_F_hat, Gamma0_hat, Gamma1_hat, lambda_0, epsilon, verbose = True)
A_hat = xp.diag(D_hat.ravel(order = 'F')) + xp.kron(A_F_hat, xp.eye(N, dtype = dtype)) + xp.kron(xp.eye(F, dtype = dtype), A_N_hat)   
m = (t+1) % M
b_hat_next = x_umean[:,m] - A_hat.dot(x_lmean[:,m-1])
        
# Prepare the input of homotopy algorithms and the regularization parameter tuning procedure
dim_DF = int(N*F + .5*F*(F-1))
iuN = triu_indices(N)
iuF = triu_indices(F)
nz = np.where(A_N_hat[iuN] != 0)[0]
z = np.where(A_N_hat[iuN] == 0)[0]
wN1 = xp.sign(A_N_hat[iuN][nz])
KN1 = (nz + dim_DF).tolist()
K1 = list(range(dim_DF)) + KN1
w1 = xp.concatenate((xp.zeros(dim_DF, dtype),wN1))
iGamma01 = np.linalg.inv(Gamma0[np.ix_(K1,K1)])

# Initialize the error metrics
lambda_max = xp.float64(0.5).astype(dtype) 
lambda_old = lambda_0
lambda_evol = []
pred_err = []
pred_err_wald = []
run_time = []
run_time_wald = []

# Online iterations (start firstly for Approach 2)
while t <= (dim-1):
    print(t)
    m = (t+1) % M # current month      
    x_old_obs = dt_MTS[:,:,int(t)].reshape(-1, 1, order = 'F').astype(dtype)
    x_new_obs = dt_MTS[:,:,int(t+1)].reshape(-1, 1, order = 'F').astype(dtype)

    pred_err.append(np.linalg.norm(A_hat.dot(x_old_obs) + b_hat_next - x_new_obs)/np.linalg.norm(x_new_obs))
    pred_err_wald.append(xp.nan)
    
    # approach 2: 
    tic = time.time()
    
    # step 1: update lambda using SGD 
    G_b = A_hat.dot(x_old_obs) + b_hat_next.reshape(-1,1) - x_new_obs
    lambda_new = lambda_update_b(N, F, KN1, iGamma01, wN1, G_b, x_old_obs, x_lmean[:,m-1], lambda_old, lambda_max, eta)
    lambda_evol.append(lambda_new)

    if lambda_old != (1+1/t)*lambda_new:
    # step 2: change lambda_old to (1+1/t)*lambda_new in the old problem ((1+1/t)*: for changing lambda and the normalization constant before data term at the same time)
    ## Homotopy algorithm 1
        _, KN1, iGamma01, wN1, lambda_trans = lambda_path(N, F, Gamma0, gamma1, KN1, wN1, lambda_old, (1+1/t)*lambda_new, iGamma01)
    
    # step 3: keep lambda_new unchanged in the problem, add new sample
    ## Homotopy algorithm 2    
    x_new_c = x_new_obs - x_lmean[:,m].reshape(-1,1)
    X_old_c = node_mat(N, F, x_old_obs.flatten() - x_lmean[:,m-1])
    as1, KN1, iGamma01, wN1, Gamma0, gamma1, mu_trans = new_data_path(N, F, iGamma01, KN1, wN1, Gamma0, gamma1, lambda_new, x_new_c, X_old_c, t, p_m[m])
    
    ## retrieve matrices from their vectorized version
    D_hat, A_N_hat, A_F_hat = as1_A(N, F, as1, KN1)
    A_hat = np.diag(D_hat.flatten(order = 'F')) + np.kron(A_F_hat, np.eye(N)) + np.kron(np.eye(F), A_N_hat)    
    
    toc = time.time()
    run_time.append(toc-tic)
    run_time_wald.append(xp.nan)
    
    Gamma0_hat = t*Gamma0_hat/(t+1) + p_m[m]*np.dot(x_old_obs - x_lmean[:,m-1].reshape(-1,1), x_old_obs.T - x_lmean[:,m-1].reshape(1,-1))/(t+1)/(p_m[m]+1)
    Gamma1_hat = t*Gamma1_hat/(t+1) + p_m[m]*np.dot(x_new_obs - x_lmean[:,m].reshape(-1,1), x_old_obs.T - x_lmean[:,m-1].reshape(1,-1))/(t+1)/(p_m[m]+1)
               
    t += 1   
    p_m[m] = p_m[m] + 1    
    x_lmean[:,m-1] = (p_m[m]-1)*x_lmean[:,m-1]/p_m[m] + x_old_obs.flatten()/p_m[m]
    m = (t+1) % M # next month    
    b_hat_next = x_lmean[:,m] - A_hat.dot(x_lmean[:,m-1])
    
    lambda_old = lambda_new

# Start Approach 1
Gamma0_hat_inv = xp.linalg.inv(Gamma0_hat) #initialize the inverse
A_ls = Gamma1_hat.dot(Gamma0_hat_inv)

A_N_hat_wald = proj_N(A_ls, N = N, F = F)
A_F_hat_wald = proj_F(A_ls, N = N, F = F)
    
# Wald test
Sigma_hat = Gamma0_hat - Gamma1_hat.dot(Gamma0_hat_inv).dot(Gamma1_hat.T)
A_N_hat_wald, _, _ = wald_test(n = t, A_N_hat = A_N_hat_wald, Gamma0_hat_inv = Gamma0_hat_inv, Sigma_hat = Sigma_hat)
A_hat_wald = Kron_str(xp.diag(A_ls).reshape(N,F,order = 'F'), A_N_hat_wald, A_F_hat_wald)
m = (t+1) % M
b_hat_next_wald = x_lmean[:,m] - A_ls.dot(x_lmean[:,m-1])

# Online update
while t <= T-2:
    print(t+1)
    m = (t+1) % M # current month    
    x_old_obs = dt_MTS[:,:,int(t)].reshape(-1, 1, order = 'F').astype(dtype)
    x_new_obs = dt_MTS[:,:,int(t+1)].reshape(-1, 1, order = 'F').astype(dtype)

    pred_err.append(np.linalg.norm(A_hat.dot(x_old_obs) + b_hat_next - x_new_obs)/np.linalg.norm(x_new_obs))
    pred_err_wald.append(np.linalg.norm(A_hat_wald.dot(x_old_obs) + b_hat_next_wald - x_new_obs)/np.linalg.norm(x_new_obs))
    
    # approach 2: 
    tic = time.time()
    
    # step 1: update lambda using SGD 
    G_b = A_hat.dot(x_old_obs) + b_hat_next.reshape(-1,1) - x_new_obs
    lambda_new = lambda_update_b(N, F, KN1, iGamma01, wN1, G_b, x_old_obs, x_lmean[:,m-1], lambda_old, lambda_max, eta)
    lambda_evol.append(lambda_new)

    if lambda_old != (1+1/t)*lambda_new:
    # step 2: change lambda_old to (1+1/t)*lambda_new in the old problem ((1+1/t)*: for changing lambda and the normalization constant before data term at the same time)
    ## Homotopy algorithm 1
        _, KN1, iGamma01, wN1, lambda_trans = lambda_path(N, F, Gamma0, gamma1, KN1, wN1, lambda_old, (1+1/t)*lambda_new, iGamma01)
    
    # step 3: keep lambda_new unchanged in the problem, add new sample
    ## Homotopy algorithm 2    
    x_new_c = x_new_obs - x_lmean[:,m].reshape(-1,1)
    X_old_c = node_mat(N, F, x_old_obs.flatten() - x_lmean[:,m-1])
    as1, KN1, iGamma01, wN1, Gamma0, gamma1, mu_trans = new_data_path(N, F, iGamma01, KN1, wN1, Gamma0, gamma1, lambda_new, x_new_c, X_old_c, t, p_m[m])
    
    ## retrieve matrices from their vectorized version
    D_hat, A_N_hat, A_F_hat = as1_A(N, F, as1, KN1)
    A_hat = np.diag(D_hat.flatten(order = 'F')) + np.kron(A_F_hat, np.eye(N)) + np.kron(np.eye(F), A_N_hat)    
    
    toc = time.time()
    run_time.append(toc-tic)
    
    
    # approach 1: 
    tic = time.time()
    
    Gamma0_hat = t*Gamma0_hat/(t+1) + p_m[m]*np.dot(x_old_obs - x_lmean[:,m-1].reshape(-1,1), x_old_obs.T - x_lmean[:,m-1].reshape(1,-1))/(t+1)/(p_m[m]+1)
    Gamma1_hat = t*Gamma1_hat/(t+1) + p_m[m]*np.dot(x_new_obs - x_lmean[:,m].reshape(-1,1), x_old_obs.T - x_lmean[:,m-1].reshape(1,-1))/(t+1)/(p_m[m]+1)

    Gamma0_hat_inv = Gamma0_hat_inv*(t+1)/t
    scale = t*(1+(1/p_m[m])) + np.dot(x_old_obs.T - x_lmean[:,m-1].reshape(-1,1).T, Gamma0_hat_inv.dot(x_old_obs - x_lmean[:,m-1].reshape(-1,1)))
    scale = t*scale/(t+1)
    Gamma0_hat_inv = Gamma0_hat_inv - \
        Gamma0_hat_inv.dot(x_old_obs - x_lmean[:,m-1].reshape(-1,1)).dot(x_old_obs.T - x_lmean[:,m-1].reshape(-1,1).T).dot(Gamma0_hat_inv)/scale
    
    A_ls = Gamma1_hat.dot(Gamma0_hat_inv)          
    A_N_hat_wald = proj_N(A_ls, N = N, F = F)
    A_F_hat_wald = proj_F(A_ls, N = N, F = F)
    
    # Wald test
    Sigma_hat = Gamma0_hat - Gamma1_hat.dot(Gamma0_hat_inv).dot(Gamma1_hat.T)
    A_N_hat_wald, _, _ = wald_test(n = t+1, A_N_hat = A_N_hat_wald, Gamma0_hat_inv = Gamma0_hat_inv, Sigma_hat = Sigma_hat)
    A_hat_wald = xp.diag(np.diag(A_ls)) + xp.kron(A_F_hat_wald, xp.eye(N, dtype = dtype)) + xp.kron(xp.eye(F, dtype = dtype), A_N_hat_wald)   
           
    toc = time.time()
    run_time_wald.append(toc-tic)
           
    t += 1   
    p_m[m] = p_m[m] + 1    
    
    x_lmean[:,m-1] = (p_m[m]-1)*x_lmean[:,m-1]/p_m[m] + x_old_obs.flatten()/p_m[m]
    m = (t+1) % M # next month    
    b_hat_next = x_lmean[:,m] - A_hat.dot(x_lmean[:,m-1])
    b_hat_next_wald = x_lmean[:,m] - A_ls.dot(x_lmean[:,m-1])
    
    lambda_old = lambda_new
    
    if int(t) == int(T/3):
        print("save the graphs and trends at the {}-th iteration.".format(t))
        A_hat_1third = A_hat.copy()
        A_hat_wald_1third = A_hat_wald.copy()
        x_lmean_1third = x_lmean.copy()
    if int(t) == int(2*T/3):
        print("save the graphs and trends at the {}-th iteration.".format(t))
        A_hat_2third = A_hat.copy()
        A_hat_wald_2third = A_hat_wald.copy()
        x_lmean_2third = x_lmean.copy()


# Projected OLS
A_ls = Gamma1_hat.dot(Gamma0_hat_inv)
A_N_hat_wald = proj_N(A_ls, N = N, F = F)
A_F_hat_wald = proj_F(A_ls, N = N, F = F)
    
# Wald test
Sigma_hat = Gamma0_hat - Gamma1_hat.dot(Gamma0_hat_inv).dot(Gamma1_hat.T)
A_N_hat_wald, _, _ = wald_test(n = t, A_N_hat = A_N_hat_wald, Gamma0_hat_inv = Gamma0_hat_inv, Sigma_hat = Sigma_hat)
A_hat_wald = Kron_str(xp.diag(A_ls).reshape(N,F,order = 'F'), A_N_hat_wald, A_F_hat_wald)

# Visualization
rdgn = sns.diverging_palette(h_neg=240, h_pos=10, s=99, l=55, sep=3, as_cmap=True)
plt.figure()
ax = plt.axes()
heatmap(A_N_hat, center = 0.0, cmap = rdgn, ax = ax)
ax.set_title("Lasso")
plt.figure()
heatmap(A_N_hat_wald, center = 0.0, cmap = rdgn)

plt.figure()
heatmap(A_F_hat,  center = 0.0, cmap = rdgn)
plt.figure()
heatmap(A_F_hat_wald, center = 0.0, cmap = rdgn)

plt.figure()
plt.plot(x_lmean[0,:],  label = "tmin")
plt.plot(x_lmean[N,:],  label = "tmax")
plt.plot(x_lmean[3*N,:],  label = "prcp")
plt.legend()


# find the common signed edges 
a = (A_N_hat > 0) + -1*(A_N_hat < 0)
b = (A_N_hat_wald > 0) + -1*(A_N_hat_wald < 0)
c = a*b
adj = a*c + b*c

plt.figure()
heatmap(adj, center = 0.0, cmap = rdgn)

plt.figure()
plt.plot(lambda_evol)
plt.figure()
plt.plot(pred_err_wald, label = "Approach 1")
plt.plot(pred_err, label = "Approach 2")
plt.legend()
# the lower pred err can be considered as a justification that when 
# A_N_hat and A_N_hat_wald are different, A_N_hat is more reliable 


# save results
params = {}
params['N'] = N
params['F'] = F
params['T'] = 20
params['dtype'] = dtype
params['eta'] = eta
params['lambda_0'] = lambda_0
params['epsilon'] = epsilon

with open('results/real_data_aug','wb') as fp:
    pickle.dump((A_N_hat, A_N_hat_wald, A_F_hat, A_F_hat_wald, \
                 A_hat, A_hat_wald, A_ls, x_lmean, \
                 A_hat_1third, A_hat_wald_1third,x_lmean_1third, \
                 A_hat_2third, A_hat_wald_2third,x_lmean_2third, \
                 pred_err, pred_err_wald, lambda_evol, \
                 run_time, run_time_wald, params, \
                 ind, station_name),fp)


