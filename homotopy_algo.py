#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yiyejiang
"""

import numpy as np
import pywt
from utilities import *
from sklearn import datasets
import time
import os

if os.environ['GPU_use'] == 'False':
    import numpy as xp
    from numpy import linalg as lg
    isgpu = False
else:
    import cupy as xp
    from cupy import linalg as lg
    isgpu = True

# The codes are created based on Github repo: pierreg/reclasso 
# which is the implementation of paper :
# P. Garrigues and L. El Ghaoui, An Homotopy Algorithm for the Lasso with Online Observations, in Advances in Neural Information Processing Systems 21 (NIPS 2008).


# Accelerated Proximal Gradient Descent with backtracking line search to obtain the batch solution
def PGD_gl(D_init, A_N_init, A_F_init, Gamma0_hat, Gamma1_hat, lda, epsilon, beta = 0.5, niter_max = 5000, verbose = True):
    assert D_init.dtype == A_N_init.dtype == A_F_init.dtype == Gamma0_hat.dtype == Gamma1_hat.dtype, "Data types of the inputs are not identical."
    dtype = A_N_init.dtype
    lda = xp.float64(lda).astype(dtype)
    beta = xp.float64(beta).astype(dtype)
    
    N = A_N_init.shape[0]
    F = A_F_init.shape[0]
    
    D = D_init.copy()
    A_N = A_N_init.copy()
    A_F = A_F_init.copy()
    
    D_m1 = D.copy()
    A_N_m1 = A_N.copy()
    A_F_m1 = A_F.copy()
    
    D_m2 = D.copy()
    A_N_m2 = A_N.copy()
    A_F_m2 = A_F.copy()
    
    eta = xp.float64(1).astype(dtype)
    for n in range(1,niter_max): 
        omega = xp.float64(n/(n+3)).astype(dtype)
        D_tilde = D_m1 + omega*(D_m1 - D_m2)
        A_N_tilde = A_N_m1 + omega*(A_N_m1 - A_N_m2)
        A_F_tilde = A_F_m1 + omega*(A_F_m1 - A_F_m2)
        
        while True:       
            # calculate the gradient of smooth part
            A_tilde = Kron_str(D_tilde, A_N_tilde, A_F_tilde)
            grad = A_tilde.dot(Gamma0_hat) - Gamma1_hat
            grad_D = xp.diag(grad).reshape(N,F, order = 'F')
            grad_N = proj_N(grad,N,F)
            grad_F = proj_F(grad,N,F)
            
            # project gradient to the space K and do the gradient descent           
            D = D_tilde - eta*grad_D
            A_N = A_N_tilde - eta*grad_N
            A_F = A_F_tilde - eta*grad_F
            
            # pass to the proximal operator of the non smooth term       
            if isgpu: 
                A_N = xp.asarray(pywt.threshold(A_N.get(), eta*lda*F, mode='soft', substitute=0))
            else:
                A_N = xp.asarray(pywt.threshold(A_N, eta*lda*F, mode='soft', substitute=0))
                
            A = Kron_str(D, A_N, A_F) 
            c = xp.float64(0.5).astype(dtype)
            issmaller = c*xp.trace(A.dot(A).dot(Gamma0_hat)) - c*xp.trace(A_tilde.dot(A_tilde).dot(Gamma0_hat)) \
                - xp.trace(Gamma1_hat.dot(A-A_tilde)) - xp.trace(Kron_str(grad_D, grad_N, grad_F).dot(A-A_tilde))\
                - c*lg.norm(A-A_tilde,'fro')**2/eta <= 0
            if issmaller: break
            eta *= beta
            
        # convergence check        
        RMSD = xp.mean(xp.asarray([lg.norm(D_m1 - D), lg.norm(A_N_m1 - A_N), lg.norm(A_F_m1 - A_F)]))
        if verbose: print(RMSD)
        if RMSD < epsilon: break
    
        D_m2 = D_m1.copy()
        A_N_m2 = A_N_m1.copy()
        A_F_m2 = A_F_m1.copy()
        
        D_m1 = D.copy()
        A_N_m1 = A_N.copy()
        A_F_m1 = A_F.copy()
    if n == niter_max-1: print("Maximal iteration numbers has reached.")
    return D, A_N, A_F, n



# helper functions
# Convertions between vectors and matrices 
def as1_A(N, F, as1, KN1):
    """From as1 to D, A_N, A_F"""
    N = int(N)
    F = int(F)
    if as1.dtype == xp.float32:
        cF = xp.float32(2*N)
        cN = xp.float32(2*F)
    elif as1.dtype == xp.float64:
        cF = xp.float32(2*N)
        cN = xp.float32(2*F)        
    
    A_F = xp.zeros(shape = (F,F), dtype = as1.dtype)
    iuF = triu_indices(F)
    A_F[iuF] = as1[N*F:(N*F+len(iuF[0]))]/cF
    A_F[(iuF[1], iuF[0])] = as1[N*F:(N*F+len(iuF[0]))]/cF
    
    A_N = xp.zeros(shape = (N,N), dtype = as1.dtype)
    if len(KN1) > 0: 
        iuN = triu_indices(N)
        nz = np.array(KN1) - int(N*F + .5*F*(F-1))
        A_N[(iuN[0][nz], iuN[1][nz])] = as1[-len(nz):]/cN
        A_N[(iuN[1][nz], iuN[0][nz])] = as1[-len(nz):]/cN
    
    return as1[:N*F], A_N, A_F

def A_a1(N, F, A, KN1):
    """From A to a1 according to the order in KN1"""
    N = int(N)
    F = int(F)
    
    A_N = proj_N(A, N = N, F = F)
    A_F = proj_F(A, N = N, F = F)
    
    dim_DF = int(N*F + .5*F*(F-1))
    a1 = xp.zeros(dim_DF+len(KN1), dtype = A.dtype)
    
    a1[:N*F] = A[xp.arange(N*F), xp.arange(N*F)]
    
    iuF = triu_indices(F)
    a1[N*F:(N*F+len(iuF[0]))] = A_F[iuF]
    
    iuN = triu_indices(N)
    nz = np.array(KN1) - dim_DF
    a1[-len(nz):] = A_N[iuN][nz]
    return a1

# 
def node_mat(N, F, x):
    """generate X_t,i in the paper"""
    N = int(N)
    F = int(F)
    if x.dtype == xp.float32:
        cF = xp.float32(2*N)
        cN = xp.float32(2*F)
    elif x.dtype == xp.float64:
        cF = xp.float32(2*N)
        cN = xp.float32(2*F)     
        
    x = x.flatten()
    dim = int(N*F + .5*F*(F-1) + .5*N*(N-1))
    X = xp.zeros(shape = (dim, N*F), dtype = x.dtype)
    X[xp.arange(N*F),xp.arange(N*F)] = x
        
    iuF = triu_indices(F)
    for i in range(N*F):
        # the row indices correspond to U_kF
        for k in range(int(.5*F*(F-1))):
            r = iuF[0][k]
            s = iuF[1][k]             
            if (i // N) == r:
                X[k+N*F,i] = x[s*N + i % N]/cF
            elif (i // N) == s:
                X[k+N*F,i] = x[r*N + i % N]/cF
            else: 
                continue
        # the row indices correspond to U_kN
        iuN = triu_indices(N)
        for k in range(int(.5*N*(N-1))):
            l = iuN[0][k]
            h = iuN[1][k]             
            if (i-l) % N == 0:
                X[k+N*F+int(.5*F*(F-1)),i] = x[i-l + h]/cN
            elif (i-h) % N == 0:
                X[k+N*F+int(.5*F*(F-1)),i] = x[i-h + l]/cN
            else: 
                continue
    return X

def invupdatered(A, c):
    """update the inverse of a matrix reducing one column and one row"""
    n, m = A.shape
    indn = np.arange(n)
    q = A[c, c]
    c1 = np.hstack((indn[:c], indn[c+1:]))
    Ax = xp.atleast_2d(A[c1, [c]])
    yA = xp.atleast_2d(A[[c], c1])
    return A[c1][:,c1] - xp.dot(Ax.T, yA)/q

def invupdateapp(A, x, y, r):
    """update the inverse of a matrix appending one column and one row"""
    yA = xp.dot(y, A)
    q = 1 / (r - xp.dot(yA, x))
    Ax = q * xp.dot(A, x)
    return xp.vstack([xp.hstack([A + xp.dot(Ax, yA), -Ax]), xp.hstack([-yA * q, q])])

def Sherman_Morrison_invupdate(A, u, v):
    """update A to (A^-1 + uv)^-1 using Sherman Morrison formula"""
    return A - xp.dot(A,u).dot(v).dot(A)/(1+xp.dot(v,A).dot(u))


def lambda_path(N, F, Gamma0, gamma1, KN1, wN1, lambda_old, lambda_new, iGamma01, verbose=False):
    """
    Homotopy algorithm: A(lambda_old) -> A(lambda_new) 
    lambda_new = (1+1/t)lambda_{t+1} to avoid add u in front of the data term
    """
    assert Gamma0.dtype == gamma1.dtype == wN1.dtype == iGamma01.dtype, "Data types of the inputs are not identical."
    assert lambda_old is not lambda_new
    dtype = Gamma0.dtype
    N = xp.float64(N).astype(dtype)
    F = xp.float64(F).astype(dtype)
    
    gamma1 = gamma1.flatten()
    dim_DF = int(N*F + .5*F*(F-1))
    KN0 = np.setdiff1d(np.arange(dim_DF, dim_DF + .5*N*(N-1), dtype = int), KN1).tolist()
    lda = lambda_old
    
    lda_trans = []
    trans_type = -1
    trans_sign = 0
    trans_ind = -1
    while (lda < lambda_new)*(lambda_old < lambda_new) + (lda > lambda_new)*(lambda_old > lambda_new):
        if trans_type != -1: 
            if verbose: print(lda, trans_type, trans_ind)
            lda_trans.append((lda, trans_type, trans_ind))
        K1 = list(range(dim_DF)) + KN1
        gamma11 = gamma1[K1]
        gamma10 = gamma1[KN0]
        Gamma00 = Gamma0[np.ix_(KN0, K1)]
        w1 = xp.concatenate((xp.zeros(dim_DF, dtype),wN1))
        
        # find the transition point where non zero A_N upper diagonal element becomes zero
        if len(KN1) > 0: 
            lda_0 = xp.dot(iGamma01[-len(KN1):,:], gamma11)/xp.dot(iGamma01[-len(KN1):,:], w1)/xp.sqrt(2*F).astype(dtype)   
        else:
            lda_0 = []
        # find the transition point where zero A_N upper diagonal element becomes non zero
        M = xp.dot(Gamma00, iGamma01)
        lda_p = (gamma10 - xp.dot(M,gamma11))/(1 - xp.dot(M,w1))/xp.sqrt(2*F).astype(dtype) 
        lda_m = (gamma10 - xp.dot(M,gamma11))/(-1 - xp.dot(M,w1))/xp.sqrt(2*F).astype(dtype) 
        
        if trans_type > 0: lda_0[-1] = lambda_new
        if trans_type == 0:
            if trans_sign == 1: lda_p[np.where(np.array(KN0) == trans_ind)[0]] = lambda_new + 1
            else: lda_m[np.where(np.array(KN0) == trans_ind)[0]] = lambda_new + 1
            
        if lambda_old < lambda_new: 
            if len(lda_0) > 0: lda_0[lda_0 <= lda] = lambda_new
            if len(lda_p) > 0: lda_p[lda_p <= lda] = lambda_new
            if len(lda_m) > 0: lda_m[lda_m <= lda] = lambda_new
        else:
            if len(lda_0) > 0: lda_0[lda_0 >= lda] = lambda_new
            if len(lda_p) > 0: lda_p[lda_p >= lda] = lambda_new
            if len(lda_m) > 0: lda_m[lda_m >= lda] = lambda_new        
                
        if len(lda_0) > 0:            
            lda_0_argm = lda_0.argmin() if lambda_old < lambda_new else lda_0.argmax()
            lda_0_m = lda_0[lda_0_argm]
        else: 
            lda_0_m = lambda_new
            
        if len(lda_p) > 0:
            lda_p_argm = lda_p.argmin() if lambda_old < lambda_new else lda_p.argmax()
            lda_p_m = lda_p[lda_p_argm]
        else: 
            lda_p_m = lambda_new
            
        if len(lda_m) > 0:
            lda_m_argm = lda_m.argmin() if lambda_old < lambda_new else lda_m.argmax()
            lda_m_m = lda_m[lda_m_argm]
        else: 
            lda_m_m = lambda_new
            
        # find the next transition point
        lda_next_all = xp.array([xp.asarray(lda_0_m), xp.asarray(lda_p_m), xp.asarray(lda_m_m)])
        trans_type = lda_next_all.argmin() if lambda_old < lambda_new else lda_next_all.argmax()
        lda_next = lda_next_all[trans_type]        
        
        if (lda_next < lambda_new)*(lambda_old < lambda_new) + (lda_next > lambda_new)*(lambda_old > lambda_new):            
            lda = lda_next
            # update active set KN1, sign vector wN1, and iGamma01
            if trans_type == 0: # a non zero element becomes zero
                if isgpu:
                    trans_ind = KN1[lda_0_argm.get()]
                    trans_sign = wN1[lda_0_argm.get()]
                else:
                    trans_ind = KN1[lda_0_argm]
                    trans_sign = wN1[lda_0_argm]                    
                KN1.remove(trans_ind)
                KN0.append(trans_ind) 
                if isgpu:
                    wN1 = xp.asarray(np.delete(wN1.get(), lda_0_argm.get()))
                    iGamma01 = invupdatered(iGamma01, c = dim_DF + lda_0_argm.get())
                else:
                    wN1 = np.delete(wN1, lda_0_argm)
                    iGamma01 = invupdatered(iGamma01, c = dim_DF + lda_0_argm)                    
            else:
                if trans_type == 1: # a zero element becomes positive 
                    if isgpu:
                        trans_ind = KN0[lda_p_argm.get()]
                    else:
                        trans_ind = KN0[lda_p_argm]
                    wN1 = xp.concatenate((wN1, xp.array([1],dtype=dtype)))
                    iGamma01 = invupdateapp(iGamma01, Gamma0[K1,trans_ind].reshape(-1,1), \
                            Gamma0[trans_ind, K1].reshape(1,-1), Gamma0[trans_ind,trans_ind]) 
                    KN1.append(trans_ind)
                    KN0.remove(trans_ind) 
                else: # a zero element becomes negative
                    if isgpu:
                        trans_ind = KN0[lda_m_argm.get()]
                    else:
                        trans_ind = KN0[lda_m_argm]
                    wN1 = xp.concatenate((wN1, xp.array([-1],dtype=dtype)))
                    iGamma01 = invupdateapp(iGamma01, Gamma0[K1,trans_ind].reshape(-1,1), \
                            Gamma0[trans_ind, K1].reshape(1,-1), Gamma0[trans_ind,trans_ind]) 
                    KN1.append(trans_ind)
                    KN0.remove(trans_ind)                                    
        else:
            as1 = xp.dot(iGamma01, gamma11 - xp.sqrt(2*F).astype(dtype)*lambda_new*w1)
            lda = lambda_new
    return as1, KN1, iGamma01, wN1, lda_trans 
                
    
def new_data_path(N, F, iGamma01, KN1, wN1, Gamma0, gamma1, lambda_new, x_new,\
                  X_old, t, t_M = None, verbose=False):
    """
    Homotopy algorithm: A(t) -> A(t+1) 
    """        
    assert Gamma0.dtype == gamma1.dtype == wN1.dtype == lambda_new.dtype == x_new.dtype == X_old.dtype == iGamma01.dtype, "Data types of the inputs are not identical."
    dtype = Gamma0.dtype
    t = xp.float64(t).astype(dtype)
    N = xp.float64(N).astype(dtype)
    F = xp.float64(F).astype(dtype)
    
    # if the model considers the monthly means, if yes 
    # the input x_new, X_old are already substracted upper mean and lower mean respectively
    if t_M is not None:
        isbias = True
        t_M = xp.float64(t_M).astype(dtype)
    else:
        isbias = False
    nconst = t_M/t/(t_M+1) if isbias else 1/t
    gamma1 = gamma1.flatten()
    dim_DF = int(N*F + .5*F*(F-1))
    KN0 = np.setdiff1d(np.arange(dim_DF, dim_DF + .5*N*(N-1), dtype = int), KN1).tolist()
    #as1_ = as1
    mu_trans = []
    for i in range(int(N*F)):
        mu = 0
        trans_type = -1
        trans_sign = 0
        trans_ind = -1
        while mu < 1:
            if trans_type != -1: 
                if verbose: print(i, mu, trans_type, trans_ind)
                mu_trans.append((i, mu, trans_type, trans_ind))
            
            K1 = list(range(dim_DF)) + KN1
            gamma11 = gamma1[K1]
            gamma10 = gamma1[KN0]
            Gamma00 = Gamma0[np.ix_(KN0, K1)]
            w1 = xp.concatenate((xp.zeros(dim_DF, wN1.dtype),wN1))
            x1_old_i = X_old[K1,i].flatten()
            x0_old_i = X_old[KN0,i].flatten()
            
            as1_ = xp.dot(iGamma01, gamma11 - (1+1/t)*xp.sqrt(2*F).astype(dtype)*lambda_new*w1)
            e = x_new[i] - xp.dot(x1_old_i.T, as1_)
            u = xp.dot(iGamma01, x1_old_i)
            alpha = xp.dot(x1_old_i.T, u)
            if isbias: alpha += t/t_M # considering the monthly bias in this case
            
            # find the transition point where non zero A_N upper diagonal element becomes zero            
            if len(KN1) > 0:
                mu_0 = -t*as1_[-len(KN1):] / (alpha*as1_[-len(KN1):] + e*u[-len(KN1):])         
            else:
                mu_0 = []
            # find the transition point where zero A_N upper diagonal element becomes non zero
            b_p = xp.dot(Gamma00,as1_) - gamma10 + (1+1/t)*xp.sqrt(2*F).astype(dtype)*lambda_new
            b_m = xp.dot(Gamma00,as1_) - gamma10 - (1+1/t)*xp.sqrt(2*F).astype(dtype)*lambda_new
            mu_p = -t*b_p / (e*xp.dot(Gamma00,u) - e*x0_old_i + alpha*b_p)
            mu_m = -t*b_m / (e*xp.dot(Gamma00,u) - e*x0_old_i + alpha*b_m)      
            
            if trans_type > 0: mu_0[-1] = 1.
            if trans_type == 0:
                if trans_sign == 1: mu_p[np.where(np.array(KN0) == trans_ind)[0]] = 1.
                else: mu_m[np.where(np.array(KN0) == trans_ind)[0]] = 1.
        
            if len(mu_0) > 0: mu_0[mu_0 <= mu] = 1.
            if len(mu_p) > 0: mu_p[mu_p <= mu] = 1.
            if len(mu_m) > 0: mu_m[mu_m <= mu] = 1.
                         
            if len(mu_0) > 0:            
                mu_0_argm = mu_0.argmin() 
                mu_0_m = mu_0[mu_0_argm]
            else: 
                mu_0_m = 1.
                
            if len(mu_p) > 0:
                mu_p_argm = mu_p.argmin() 
                mu_p_m = mu_p[mu_p_argm]
            else: 
                mu_p_m = 1.
                
            if len(mu_m) > 0:
                mu_m_argm = mu_m.argmin() 
                mu_m_m = mu_m[mu_m_argm]
            else: 
                mu_m_m = 1.
                
            # find the next transition point
            mu_next_all = xp.array([xp.asarray(mu_0_m), xp.asarray(mu_p_m), xp.asarray(mu_m_m)])
            trans_type = mu_next_all.argmin() 
            mu_next = mu_next_all[trans_type]
            
            if mu_next < 1:
                mu = mu_next
                if trans_type == 0: # a non zero element becomes zero 
                    if isgpu:
                        trans_ind = KN1[mu_0_argm.get()]
                        trans_sign = wN1[mu_0_argm.get()]
                    else:
                        trans_ind = KN1[mu_0_argm]
                        trans_sign = wN1[mu_0_argm]
                    KN1.remove(trans_ind)
                    KN0.append(trans_ind)            
                    if isgpu:
                        wN1 = xp.asarray(np.delete(wN1.get(), mu_0_argm.get()))
                        iGamma01 = invupdatered(iGamma01, c = dim_DF + mu_0_argm.get())
                    else:
                        wN1 = np.delete(wN1, mu_0_argm)
                        iGamma01 = invupdatered(iGamma01, c = dim_DF + mu_0_argm)                       
                else:
                    if trans_type == 1: # a zero element becomes positive 
                        if isgpu:
                            trans_ind = KN0[mu_p_argm.get()]
                        else:
                            trans_ind = KN0[mu_p_argm]
                        wN1 = xp.concatenate((wN1, xp.array([1],dtype=xp.float32)))
                        iGamma01 = invupdateapp(iGamma01, Gamma0[K1,trans_ind].reshape(-1,1), \
                                Gamma0[trans_ind, K1].reshape(1,-1), Gamma0[trans_ind,trans_ind])                    
                        KN1.append(trans_ind)
                        KN0.remove(trans_ind) 
                    else: # a zero element becomes negative
                        if isgpu: 
                            trans_ind = KN0[mu_m_argm.get()]
                        else:
                            trans_ind = KN0[mu_m_argm]
                        wN1 = xp.concatenate((wN1, xp.array([-1],dtype=xp.float32)))
                        iGamma01 = invupdateapp(iGamma01, Gamma0[K1,trans_ind].reshape(-1,1), \
                                Gamma0[trans_ind, K1].reshape(1,-1), Gamma0[trans_ind,trans_ind])                    
                        KN1.append(trans_ind)
                        KN0.remove(trans_ind)     
            else:                 
                mu = 1
        iGamma01 = Sherman_Morrison_invupdate(iGamma01, nconst*x1_old_i.reshape(-1,1), x1_old_i.reshape(1,-1))
        Gamma0 += nconst*xp.dot(X_old[:,i].reshape(-1,1), X_old[:,i].reshape(1,-1))
        gamma1 += nconst*x_new[i]*X_old[:,i].flatten()
    as1 = as1_ + e*u/(t+alpha)
    iGamma01 *= (1+1/t)
    Gamma0 *= t/(t+1)
    gamma1 *= t/(t+1)
    return as1, KN1, iGamma01, wN1, Gamma0, gamma1, mu_trans


def lambda_update(N, F, KN1, iGamma01, wN1, G, lambda_old, lambda_max, eta):
    """
    update lambda using stochastic gradient descent, 
    in the direction that minimizes the prediction error of the unseen data x_new
    measured by MSE 
    
    model which does not consider bias
    
    G := (A_old x_old - x_new) x_old.T
    """
    assert wN1.dtype == G.dtype == iGamma01.dtype, "Data types of the inputs are not identical."
    dtype = G.dtype
    N = xp.float64(N).astype(dtype)
    F = xp.float64(F).astype(dtype)
    
    if len(KN1) > 0: 
        a1G = A_a1(N, F, G, KN1)
        dim_DF = int(N*F + .5*F*(F-1))
        w1 = xp.concatenate((xp.zeros(dim_DF, wN1.dtype),wN1))
        grad_lda = -xp.sqrt(2*F).astype(dtype)*xp.dot(a1G,iGamma01).dot(w1)
    else:
        grad_lda = xp.float64(0).astype(dtype)
    return xp.clip(lambda_old - eta*grad_lda,xp.float64(1e-5).astype(dtype),lambda_max)


def lambda_update_b(N, F, KN1, iGamma01, wN1, \
                    G_b, x_old, x_old_lmean,
                    lambda_old, lambda_max, eta
                    ):
    """
    update lambda using stochastic gradient descent, 
    in the direction that minimizes the prediction error of the unseen data x_new
    measured by MSE 
    
    model which considers bias
    """
    assert wN1.dtype == G_b.dtype == x_old.dtype ==x_old_lmean.dtype == iGamma01.dtype, "Data types of the inputs are not identical."
    dtype = G_b.dtype
    N = xp.float64(N).astype(dtype)
    F = xp.float64(F).astype(dtype)
    
    if len(KN1) > 0: 
        G = xp.dot(G_b,x_old.reshape(1,-1))  
        
        a1G = A_a1(N, F, G, KN1)
        dim_DF = int(N*F + .5*F*(F-1))
        w1 = xp.concatenate((xp.zeros(dim_DF, wN1.dtype),wN1))
        das1_lda = -xp.sqrt(2*F).astype(dtype)*xp.dot(iGamma01, w1)
        grad_lda_A = xp.dot(a1G,das1_lda).flatten()
            
        dD_lda, dA_N_lda, dA_F_lda = as1_A(N, F, das1_lda, KN1)
        dA_lda = xp.diag(dD_lda) + xp.kron(dA_F_lda, xp.eye(int(N), dtype=dtype)) + xp.kron(xp.eye(int(F), dtype = dtype), dA_N_lda)
        db_lda = -xp.dot(dA_lda, x_old_lmean.reshape(-1,1))
        grad_lda_b = xp.dot(G_b.reshape(1,-1), db_lda).flatten()
    else:
        grad_lda_A = grad_lda_b = 0     
    return xp.clip(lambda_old - eta*(grad_lda_b + grad_lda_A),1e-5,lambda_max)


def test_onlinepgl(N = 10, F = 5, nnz = 7, T = 2, eta = 1e-5, nupdate = 500, nsim = 1, \
                   dtype = xp.float64, lambda_0 = 0.02, vabs_min = 0.1, verbose=False):
    # prepare the error metric 
    N = int(N)
    F = int(F)
    dim = int(N*F + .5*F*(F-1) + .5*N*(N-1)) # Dimension of space K
    
    est_err = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    est_err_wald = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    pred_err = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    pred_err_wald = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    lambda_evol = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    run_time = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    run_time_wald = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    
    for j in range(nsim):    
        if verbose:
            if j == 0:
                print("Running the 1st simulation.")
            elif j == 1:
                print("Running the 2nd simulation.")
            elif j == 2:
                print("Running the 3rd simulation.")
            else:                
                print("Running the {}th simulation.".format(j+1))
            
        # generate random product graph
        D, A_N, A_F, Ax = generator_rg(N, F, nnz, dtype = dtype, vabs_min = vabs_min)
              
        Sigma = xp.asarray(datasets.make_spd_matrix(N*F), dtype = dtype)# noise covariance
        
        # Batch problem setup
        Gamma0_hat = xp.zeros(shape = (N*F, N*F), dtype = dtype)
        Gamma1_hat = xp.zeros(shape = (N*F, N*F), dtype = dtype)
        
        Gamma0 = xp.zeros(shape = (dim, dim), dtype = dtype) # The structured Gamma0_hat used in HomoAlgos
        gamma1 = xp.zeros(shape = (dim, 1), dtype = dtype) # The structured Gamma1_hat used in HomoAlgos
        
        x_old = xp.random.normal(size = (N*F, 1)).astype(dtype) # Initial signal
        for t in xp.arange(1, T + 1, dtype = xp.float32):
            x_new = syn(x_old, Ax, Sigma)  
            
            Gamma0_hat = (t-1)*Gamma0_hat/t + x_old.dot(x_old.T)/t
            Gamma1_hat = (t-1)*Gamma1_hat/t + x_new.dot(x_old.T)/t
            
            # structured Gamma0_hat, Gamma1_hat prepared for the online algos
            X_old = node_mat(N, F, x_old)
            Gamma0 = (t-1)*Gamma0/t + X_old.dot(X_old.T)/t
            gamma1 = (t-1)*gamma1/t + X_old.dot(x_new)/t
            x_old = x_new.copy()
            
        # Start the online procedure 2
        # PGD solve batch problem
        D_hat, A_N_hat, A_F_hat = xp.zeros(shape = (N,F), dtype = dtype), xp.zeros(shape = (N,N), dtype = dtype), xp.zeros(shape = (F,F), dtype = dtype)
        lambda_0 = xp.float64(lambda_0).astype(dtype)
        epsilon = xp.float64(1e-5).astype(dtype)
        D_hat, A_N_hat, A_F_hat, n = PGD_gl(D_hat, A_N_hat, A_F_hat, Gamma0_hat, Gamma1_hat, lambda_0, epsilon, verbose = verbose)
        A_hat = xp.diag(D_hat.ravel(order = 'F')) + xp.kron(A_F_hat, xp.eye(N, dtype = dtype)) + xp.kron(xp.eye(F, dtype = dtype), A_N_hat)   
            
        # Prepare the input of homotopy algorithms and the SGD
        dim_DF = int(N*F + .5*F*(F-1))
        iuN = triu_indices(N)
        nz = np.where(A_N_hat[iuN] != 0)[0]
        wN1 = xp.sign(A_N_hat[iuN][nz])
        KN1 = (nz + dim_DF).tolist()
        K1 = list(range(dim_DF)) + KN1
        iGamma01 = xp.linalg.inv(Gamma0[np.ix_(K1,K1)])
        
        eta = xp.float64(eta).astype(dtype) # step size for lambda updating
        lambda_max = xp.float64(0.1).astype(dtype)
        lambda_old = lambda_0

        # Online iterations 
        while t <= dim:
            x_new = syn(x_old, Ax, Sigma)
        
            pred_err[j, int(t) - T] = xp.linalg.norm(A_hat.dot(x_old) - x_new)/xp.linalg.norm(x_new)
            pred_err_wald[j, int(t) - T] = xp.nan
        
            # approach 2: 
            tic = time.time()
        
            # step 1: update lambda using SGD 
            G = xp.dot(A_hat.dot(x_old) - x_new, x_old.T)
            lambda_new = lambda_update(N, F, KN1, iGamma01, wN1, G, lambda_old, lambda_max, eta)
            #print("New lambda: {}\n".format(lambda_new))
            lambda_evol[j, int(t) - T] = lambda_new
        
            if lambda_old != xp.float64(1+1/t).astype(dtype)*lambda_new:
            # step 2: change lambda_old to (1+1/t)*lambda_new in the old problem ((1+1/t)*: for changing lambda and the normalization constant before data term at the same time)
            ## Homotopy algorithm 1
                _, KN1, iGamma01, wN1, lambda_trans = lambda_path(N, F, Gamma0, gamma1, KN1, wN1, lambda_old, xp.float64(1+1/t).astype(dtype)*lambda_new, iGamma01)
        
            # step 3: keep lambda_new unchanged in the problem, add new sample
            ## Homotopy algorithm 2
            X_old = node_mat(N, F, x_old)
            as1, KN1, iGamma01, wN1, Gamma0, gamma1, mu_trans = new_data_path(N, F, iGamma01, KN1, wN1, Gamma0, gamma1, lambda_new, x_new, X_old, t, t_M = None)
        
            ## retrieve matrices from their vectorized version
            D_hat, A_N_hat, A_F_hat = as1_A(N, F, as1, KN1)
            A_hat = xp.diag(D_hat.ravel(order = 'F')) + xp.kron(A_F_hat, xp.eye(N, dtype = dtype)) + xp.kron(xp.eye(F, dtype = dtype), A_N_hat)    
            est_err[j, int(t) - T] = xp.linalg.norm(A_hat - Ax)/xp.linalg.norm(Ax)
            est_err_wald[j, int(t) - T] = xp.nan
        
            toc = time.time()
            run_time[j, int(t) - T] = toc-tic
            run_time_wald[j, int(t) - T] = xp.nan
        
            t += 1    
        
            Gamma0_hat = (t-1)*Gamma0_hat/t + x_old.dot(x_old.T)/t
            Gamma1_hat = (t-1)*Gamma1_hat/t + x_new.dot(x_old.T)/t
        
            lambda_old = lambda_new
            x_old = x_new.copy()                
        
        
        # Start the online procedure 1
        Gamma0_hat_inv = xp.linalg.inv(Gamma0_hat) #initialize the inverse
        A_ls = Gamma1_hat.dot(Gamma0_hat_inv)
        
        A_N_hat_wald = proj_N(A_ls, N = N, F = F)
        A_F_hat_wald = proj_F(A_ls, N = N, F = F)
            
        # Wald test
        Sigma_hat = Gamma0_hat - Gamma1_hat.dot(Gamma0_hat_inv).dot(Gamma1_hat.T)
        A_N_hat_wald, _, _ = wald_test(n = t, A_N_hat = A_N_hat_wald, Gamma0_hat_inv = Gamma0_hat_inv, Sigma_hat = Sigma_hat)
        A_hat_wald = xp.diag(xp.diag(A_ls)) + xp.kron(A_F_hat_wald, xp.eye(N, dtype = dtype)) + xp.kron(xp.eye(F, dtype = dtype), A_N_hat_wald) 

  
        # Online updates. 
        for n in range(nupdate):
            if n % 50 == 0: print(n)
            x_new = syn(x_old, Ax, Sigma)        
            pred_err_wald[j, int(t) - T] = xp.linalg.norm(A_hat_wald.dot(x_old) - x_new)/xp.linalg.norm(x_new)
            pred_err[j, int(t) - T] = xp.linalg.norm(A_hat.dot(x_old) - x_new)/xp.linalg.norm(x_new)

            # approach 2: 
            tic = time.time()
            
            # step 1: update lambda using SGD 
            G = xp.dot(A_hat.dot(x_old) - x_new, x_old.T)
            lambda_new = lambda_update(N, F, KN1, iGamma01, wN1, G, lambda_old, lambda_max, eta)
            lambda_evol[j, int(t) - T] = lambda_new
        
            if lambda_old != xp.float64(1+1/t).astype(dtype)*lambda_new:
            # step 2: change lambda_old to (1+1/t)*lambda_new in the old problem ((1+1/t)*: for changing lambda and the normalization constant before data term at the same time)
            ## Homotopy algorithm 1
                _, KN1, iGamma01, wN1, lambda_trans = lambda_path(N, F, Gamma0, gamma1, KN1, wN1, lambda_old, xp.float64(1+1/t).astype(dtype)*lambda_new, iGamma01)
            
            # step 3: keep lambda_new unchanged in the problem, add new sample
            ## Homotopy algorithm 2
            X_old = node_mat(N, F, x_old)
            as1, KN1, iGamma01, wN1, Gamma0, gamma1, mu_trans = new_data_path(N, F, iGamma01, KN1, wN1, Gamma0, gamma1, lambda_new, x_new, X_old, t, t_M = None)
            
            ## retrieve matrices from their vectorized version
            D_hat, A_N_hat, A_F_hat = as1_A(N, F, as1, KN1)
            A_hat = xp.diag(D_hat.ravel(order = 'F')) + xp.kron(A_F_hat, xp.eye(N, dtype = dtype)) + xp.kron(xp.eye(F, dtype = dtype), A_N_hat)    
            est_err[j, int(t) - T] = xp.linalg.norm(A_hat - Ax)/xp.linalg.norm(Ax)
            
            toc = time.time()
            run_time[j, int(t) - T] = toc-tic
    
            t += 1
            # approach 1: 
            tic = time.time()
            
            Gamma0_hat = (t-1)*Gamma0_hat/t + x_old.dot(x_old.T)/t
            Gamma1_hat = (t-1)*Gamma1_hat/t + x_new.dot(x_old.T)/t
            
            Gamma0_hat_inv = Gamma0_hat_inv*t/(t-1)
            scale = t + x_old.T.dot(Gamma0_hat_inv).dot(x_old)
            Gamma0_hat_inv = Gamma0_hat_inv - Gamma0_hat_inv.dot(x_old).dot(x_old.T).dot(Gamma0_hat_inv)/scale
            
            A_ls = Gamma1_hat.dot(Gamma0_hat_inv)
            
            A_N_hat_wald = proj_N(A_ls, N = N, F = F)
            A_F_hat_wald = proj_F(A_ls, N = N, F = F)
            
            # Wald test
            Sigma_hat = Gamma0_hat - Gamma1_hat.dot(Gamma0_hat_inv).dot(Gamma1_hat.T)
            A_N_hat_wald, _, _ = wald_test(n = t, A_N_hat = A_N_hat_wald, Gamma0_hat_inv = Gamma0_hat_inv, Sigma_hat = Sigma_hat)
            A_hat_wald = xp.diag(xp.diag(A_ls)) + xp.kron(A_F_hat_wald, xp.eye(N, dtype = dtype)) + xp.kron(xp.eye(F, dtype = dtype), A_N_hat_wald)   
            est_err_wald[j, int(t) - T-1] = xp.linalg.norm(A_hat_wald - Ax)/xp.linalg.norm(Ax)
            
            toc = time.time()
            run_time_wald[j, int(t) - T-1] = toc-tic
    
            lambda_old = lambda_new
            x_old = x_new.copy()
    
    return est_err, est_err_wald, pred_err, pred_err_wald, A_N, A_N_hat, A_N_hat_wald, lambda_evol, run_time, run_time_wald

    
def test_onlinepgl_aug(N = 10, F = 5, nnz = 7,  M = 12, T = 20, eta = 5e-6, nupdate = 500, nsim = 1, dtype = xp.float64, lambda_0 = 0.02, vabs_min = 0.1, verbose=False):
    
    dim = int(N*F + .5*F*(F-1) + .5*N*(N-1)) # Dimension of space K
        
    # prepare the error metric 
    est_err = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    est_err_wald = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    est_err_b = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    pred_err = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    pred_err_wald = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    lambda_evol = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    run_time = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    run_time_wald = xp.zeros(shape = (nsim, dim - T + 1 + nupdate))
    
    for j in range(nsim):   
        if verbose:
            if j == 0:
                print("Running the 1st simulation.")
            elif j == 1:
                print("Running the 2nd simulation.")
            elif j == 2:
                print("Running the 3rd simulation.")
            else:                
                print("Running the {}th simulation.".format(j+1))
        # generate random product graph
        D, A_N, A_F, Ax = generator_rg(N, F, nnz, dtype = dtype, vabs_min = vabs_min)

        b = 5*np.sin(np.arange(M)*np.math.pi/M) + np.random.normal(size = (N*F,1))
        Sigma = datasets.make_spd_matrix(N*F)

        # Batch problem setup
        Gamma0 = np.zeros(shape = (dim, dim)) # The structured Gamma0_hat used in HomoAlgos
        gamma1 = np.zeros(shape = (dim, 1)) # The structured Gamma1_hat used in HomoAlgos
        Gamma0_hat, Gamma1_hat = np.zeros(shape = (N*F, N*F)), np.zeros(shape = (N*F, N*F))
        Gamma0_b_hat, Gamma1_b_hat = np.zeros(shape = (N*F, N*F)), np.zeros(shape = (N*F, N*F))
        p_m = np.zeros(12)  
        x_lmean = np.zeros(shape = (N*F, M))
        x_umean = np.zeros(shape = (N*F, M))
        x_old = np.random.normal(size = (N*F, 1)) # Initial hidden signal 
        x_old_obs = x_old + b[:,0].reshape(-1,1) # Initial observed signal in January

        for t in np.arange(1, T + 1):
            m = t % 12 # current month
            p_m[m] = p_m[m] + 1 
            x_new = syn(x_old, Ax, Sigma)  
            x_new_obs = x_new + b[:,m].reshape(-1,1)
            
            # estimation of monthly means
            x_umean[:,m] = (p_m[m]-1)*x_umean[:,m]/p_m[m] + x_new_obs.flatten()/p_m[m]
            x_lmean[:,m-1] = (p_m[m]-1)*x_lmean[:,m-1]/p_m[m] + x_old_obs.flatten()/p_m[m]
        
            Gamma0_b_hat = (t-1)*Gamma0_b_hat/t + x_old_obs.dot(x_old_obs.T)/t
            Gamma1_b_hat = (t-1)*Gamma1_b_hat/t + x_new_obs.dot(x_old_obs.T)/t 
            
            X_old_obs = node_mat(N, F, x_old_obs)
            Gamma0 = (t-1)*Gamma0/t + X_old_obs.dot(X_old_obs.T)/t
            gamma1 = (t-1)*gamma1/t + X_old_obs.dot(x_new_obs)/t
            
            x_old = x_new.copy()
            x_old_obs = x_new_obs.copy()
            
        Gamma0_hat = Gamma0_b_hat - np.dot(x_lmean[:,np.arange(M)-1]*p_m, x_lmean[:,np.arange(M)-1].T)/t
        Gamma1_hat = Gamma1_b_hat - np.dot(x_umean*p_m, x_lmean[:,np.arange(M)-1].T)/t
        for m in range(M) :
            X_lmean = node_mat(N, F, x_lmean[:,m-1])
            Gamma0 = Gamma0 - p_m[m]*X_lmean.dot(X_lmean.T)/t
            gamma1 = gamma1 - p_m[m]*X_lmean.dot(x_umean[:,m].reshape(-1,1))/t
            
        # PGD
        D_hat, A_N_hat, A_F_hat = xp.zeros((N,F), dtype), xp.zeros((N,N), dtype), xp.zeros((F,F), dtype)
        lambda_0 = xp.float64(lambda_0).astype(dtype)
        epsilon = xp.float64(1e-5).astype(dtype)
        D_hat, A_N_hat, A_F_hat, n = PGD_gl(D_hat, A_N_hat, A_F_hat, Gamma0_hat, Gamma1_hat, lambda_0, epsilon, verbose = False)
        A_hat = xp.diag(D_hat.ravel(order = 'F')) + xp.kron(A_F_hat, xp.eye(N, dtype = dtype)) + xp.kron(xp.eye(F, dtype = dtype), A_N_hat)   
        m = (t+1) % M
        b_hat_next = x_umean[:,m] - A_hat.dot(x_lmean[:,m-1])   
        
        # Prepare the input of homotopy algorithms and the SG
        dim_DF = int(N*F + .5*F*(F-1))
        iuN = triu_indices(N)
        iuF = triu_indices(F)
        nz = np.where(A_N_hat[iuN] != 0)[0]
        wN1 = xp.sign(A_N_hat[iuN][nz])
        KN1 = (nz + dim_DF).tolist()
        K1 = list(range(dim_DF)) + KN1
        iGamma01 = np.linalg.inv(Gamma0[np.ix_(K1,K1)])

        eta = xp.float64(eta).astype(dtype) # step size for lambda updating
        lambda_max = xp.float64(0.1).astype(dtype)
        lambda_old = lambda_0
        
        # Online iterations 
        # From now on we only need <<x_lmean>>
        while t <= dim:
            m = (t+1) % M # current month    
            x_new = syn(x_old, Ax, Sigma)  
            x_new_obs = x_new + b[:,m].reshape(-1,1)
        
            pred_err[j, int(t) - T] = np.linalg.norm(A_hat.dot(x_old_obs) + b_hat_next - x_new_obs)/np.linalg.norm(x_new_obs)
            pred_err_wald[j, int(t) - T] = xp.nan
            
            # approach 2: 
            tic = time.time()
            
            # step 1: update lambda using SGD 
            G_b = A_hat.dot(x_old_obs) + b_hat_next.reshape(-1,1) - x_new_obs
            lambda_new = lambda_update_b(N, F, KN1, iGamma01, wN1, G_b, x_old_obs, x_lmean[:,m-1], lambda_old, lambda_max, eta)
            lambda_evol[j, int(t) - T] = lambda_new
        
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
            est_err[j, int(t) - T] = np.linalg.norm(A_hat - Ax)/np.linalg.norm(Ax)
            est_err_wald[j, int(t) - T] = xp.nan
            
            toc = time.time()
            run_time[j, int(t) - T] = toc-tic
            run_time_wald[j, int(t) - T] = xp.nan
            
            Gamma0_hat = t*Gamma0_hat/(t+1) + p_m[m]*np.dot(x_old_obs - x_lmean[:,m-1].reshape(-1,1), x_old_obs.T - x_lmean[:,m-1].reshape(1,-1))/(t+1)/(p_m[m]+1)
            Gamma1_hat = t*Gamma1_hat/(t+1) + p_m[m]*np.dot(x_new_obs - x_lmean[:,m].reshape(-1,1), x_old_obs.T - x_lmean[:,m-1].reshape(1,-1))/(t+1)/(p_m[m]+1)
                       
            t += 1   
            p_m[m] = p_m[m] + 1    
            x_lmean[:,m-1] = (p_m[m]-1)*x_lmean[:,m-1]/p_m[m] + x_old_obs.flatten()/p_m[m]
            m = (t+1) % M # next month    
            b_hat_next = x_lmean[:,m] - A_hat.dot(x_lmean[:,m-1])
            
            est_err_b[j, int(t) - T - 1] = np.linalg.norm(x_lmean - b)/np.linalg.norm(b)
            
            lambda_old = lambda_new
            x_old = x_new.copy()
            x_old_obs = x_new_obs.copy()
        
        
        # Start the online procedure 1
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
        
        
        # Online updates. 
        for n in range(nupdate):
            if n % 50 == 0: print(n)
            m = (t+1) % M # current month    
            x_new = syn(x_old, Ax, Sigma)  
            x_new_obs = x_new + b[:,m].reshape(-1,1)
        
            pred_err[j, int(t) - T] = np.linalg.norm(A_hat.dot(x_old_obs) + b_hat_next - x_new_obs)/np.linalg.norm(x_new_obs)
            pred_err_wald[j, int(t) - T] = np.linalg.norm(A_hat_wald.dot(x_old_obs) + b_hat_next_wald - x_new_obs)/np.linalg.norm(x_new_obs)
            
            # approach 2: 
            tic = time.time()
            
            # step 1: update lambda using SGD 
            G_b = A_hat.dot(x_old_obs) + b_hat_next.reshape(-1,1) - x_new_obs
            lambda_new = lambda_update_b(N, F, KN1, iGamma01, wN1, G_b, x_old_obs, x_lmean[:,m-1], lambda_old, lambda_max, eta)
            lambda_evol[j, int(t) - T] = lambda_new
        
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
            est_err[j, int(t) - T] = np.linalg.norm(A_hat - Ax)/np.linalg.norm(Ax)
            
            toc = time.time()
            run_time[j, int(t) - T] = toc-tic
            
            
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
            est_err_wald[j, int(t) - T] = xp.linalg.norm(A_hat_wald - Ax)/xp.linalg.norm(Ax)
                   
            toc = time.time()
            run_time_wald[j, int(t) - T] = toc-tic
                   
            t += 1   
            p_m[m] = p_m[m] + 1    
            
            x_lmean[:,m-1] = (p_m[m]-1)*x_lmean[:,m-1]/p_m[m] + x_old_obs.flatten()/p_m[m]
            m = (t+1) % M # next month    
            b_hat_next = x_lmean[:,m] - A_hat.dot(x_lmean[:,m-1])
            b_hat_next_wald = x_lmean[:,m] - A_ls.dot(x_lmean[:,m-1])
            
            est_err_b[j, int(t) - T - 1] = np.linalg.norm(x_lmean - b)/np.linalg.norm(b)
            
            lambda_old = lambda_new
            x_old = x_new.copy()
            x_old_obs = x_new_obs.copy()
    
    return est_err, est_err_wald, est_err_b, pred_err, pred_err_wald, A_N, A_N_hat, A_N_hat_wald, x_lmean, lambda_evol, run_time, run_time_wald


