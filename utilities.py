#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.signal import unit_impulse
from scipy.stats import f, chi2
import os

if os.environ['GPU_use'] == 'False':
    import numpy as xp
    from numpy import linalg as lg
    isgpu = False
else:
    import cupy as xp
    from cupy import linalg as lg
    isgpu = True
    
# Construct matrix A with the Kronecker sum structure 
def Kron_str(D, A_N, A_F):
    assert D.dtype == A_N.dtype == A_F.dtype, "Data types of the inputs are not identical."
    return xp.diag(D.ravel(order = 'F')) + xp.kron(A_F, xp.eye(A_N.shape[0], dtype = A_F.dtype)) + xp.kron(xp.eye(A_F.shape[0], dtype = A_N.dtype), A_N)

# Generate random graph A_N, A_F, and diagonal elements D
def generator_rg(N = 10, F = 5, nnz = 7, vabs_min = 0.1, dtype = xp.float32):
    N = int(N)
    F = int(F)
    
    is_notfarzero = True
    while is_notfarzero:
        A_N = xp.zeros(shape = (N,N), dtype = dtype)
        iuN = triu_indices(N)
        nz = xp.random.choice(len(iuN[0]), nnz, replace=False)
        aN1 = xp.random.uniform(1, 2, size = len(nz),).astype(dtype)*(xp.random.binomial(1, .5, len(nz)).astype(dtype)*2-1)
        A_N[(iuN[0][nz], iuN[1][nz])] = aN1
        A_N[(iuN[1][nz], iuN[0][nz])] = aN1
        
        B_F = xp.random.uniform(1, 2, size = (F,F)).astype(dtype)*(xp.random.binomial(1, .5, (F,F)).astype(dtype)*2-1)
        A_F = xp.triu(B_F, 1) + xp.tril(B_F.T, -1)
        
        D = xp.diag(xp.random.uniform(1, 2, size = N*F).astype(dtype)*(xp.random.binomial(1, .5, N*F).astype(dtype)*2-1))
        Ax = D + xp.kron(A_F, xp.eye(N, dtype = dtype)) + xp.kron(xp.eye(F, dtype = dtype), A_N)
        
        c = lg.norm(Ax,2)
        A_F /= 1.2*c
        A_N /= 1.2*c
        Ax /= 1.2*c
        
        # check there is no A_N elements too closed to 0
        if xp.abs(A_N[A_N != 0]).min() > vabs_min: is_notfarzero = False
    return D, A_N, A_F, Ax

def syn(S_old, A, Sigma):
    assert A.dtype == S_old.dtype == Sigma.dtype, "Data types of the inputs are not identical."
    P = xp.random.multivariate_normal(mean = xp.zeros(len(S_old)), cov = Sigma).astype(A.dtype)
    #P = xp.random.normal(size = (N*F, 1))
    S_new = A.dot(S_old) + xp.expand_dims(P, axis = 1)
    return S_new

def proj_N(A, N, F):
    A_N_hat = xp.zeros(shape = (N,N), dtype = A.dtype)
    for l in range(N-1):
        for h in range(l+1, N):
            u_l = xp.expand_dims(xp.asarray(unit_impulse(N, l)),axis = 1)
            u_h = xp.expand_dims(xp.asarray(unit_impulse(N, h)),axis = 1)
            loc = xp.kron(xp.eye(F), u_l.dot(u_h.T) + u_h.dot(u_l.T))
            A_N_hat += xp.trace(loc.dot(A))*(u_l.dot(u_h.T) + u_h.dot(u_l.T))/2/F
    return A_N_hat

def proj_F(A, N, F):
    A_F_hat = xp.zeros(shape = (F,F), dtype = A.dtype)
    for r in range(F-1):
        for s in range(r+1, F):
            u_r = xp.expand_dims(xp.asarray(unit_impulse(F, r)),axis = 1)
            u_s = xp.expand_dims(xp.asarray(unit_impulse(F, s)),axis = 1)
            loc = xp.kron(u_r.dot(u_s.T) + u_s.dot(u_r.T), xp.eye(N))
            A_F_hat += xp.trace(loc.dot(A))*(u_r.dot(u_s.T) + u_s.dot(u_r.T))/2/N
    return A_F_hat

def wald_test(n, A_N_hat, Gamma0_hat_inv, Sigma_hat, alpha = 0.1, is_f = True):
    """
    output: 
        A_N_hat: the sparse estimator
        row_ind, col_ind: coordinates(in upper part) of the zero weights condsiered by wald test  
    """        
    N = A_N_hat.shape[0]
    F = Gamma0_hat_inv.shape[0]//N
    
    find_psmall = xp.abs(xp.triu(A_N_hat))
    find_psmall[find_psmall == 0] = find_psmall.max() + 1
    ind = xp.argsort(find_psmall.ravel('F'))[:int(N*N/2 - N/2)]
    row_ind = (xp.array(ind) + 1) % N - 1
    row_ind = row_ind.astype(int)
    col_ind = xp.ceil((xp.array(ind) + 1)/N) - 1
    col_ind = col_ind.astype(int)
        
    s = xp.sort(find_psmall.flatten())[:int(N*N/2 - N/2)]
    s = xp.expand_dims(s, axis = 1)
    
#    Sigma_A_hat = xp.kron(Gamma0_hat_inv, Sigma_hat)
    def wald_test_oneP(P):
        CSigmaC = xp.zeros(shape = (P,P), dtype = A_N_hat.dtype)
        for p1 in range(P):
            l1 = row_ind[p1]
            h1 = col_ind[p1]
            for p2 in range(p1,P):
                l2 = row_ind[p2]
                h2 = col_ind[p2]              
                l1_mesh = xp.linspace(l1, l1 + N*(F-1), num=F, dtype = xp.int64)
                l2_mesh = xp.linspace(l2, l2 + N*(F-1), num=F, dtype = xp.int64)
                h1_mesh = xp.linspace(h1, h1 + N*(F-1), num=F, dtype = xp.int64)
                h2_mesh = xp.linspace(h2, h2 + N*(F-1), num=F, dtype = xp.int64)
                
                Sigma_hat_ll = Sigma_hat[l1_mesh,:][:,l2_mesh]
                Gamma0_hat_inv_hh = Gamma0_hat_inv[h1_mesh,:][:,h2_mesh]

                Sigma_hat_hh = Sigma_hat[h1_mesh,:][:,h2_mesh]
                Gamma0_hat_inv_ll = Gamma0_hat_inv[l1_mesh,:][:,l2_mesh]

                Sigma_hat_lh = Sigma_hat[l1_mesh,:][:,h2_mesh]
                Gamma0_hat_inv_hl = Gamma0_hat_inv[h1_mesh,:][:,l2_mesh]

                Sigma_hat_hl = Sigma_hat[h1_mesh,:][:,l2_mesh]
                Gamma0_hat_inv_lh = Gamma0_hat_inv[l1_mesh,:][:,h2_mesh]

                CSigmaC[p1,p2] = CSigmaC[p2,p1] = (xp.trace(Sigma_hat_ll.T.dot(Gamma0_hat_inv_hh)) +  \
                xp.trace(Sigma_hat_hh.T.dot(Gamma0_hat_inv_ll)) + \
                xp.trace(Sigma_hat_lh.T.dot(Gamma0_hat_inv_hl)) + \
                xp.trace(Sigma_hat_hl.T.dot(Gamma0_hat_inv_lh)))/4/F/F          
                
                #CSigmaC[p1,p2] = CSigmaC[p2,p1] = loc1.T.dot(Sigma_A_hat).dot(loc2)/4/F/F        
                                
        wald_stat = n*s[:P,0].T.dot(lg.inv(CSigmaC)).dot(s[:P,0])
        # significance level: alpha (higher the prob of commiting type 2 error decrease)
        if is_f: # do F-test
            wald_stat /= P 
            crit = f.ppf(q = 1-alpha, dfn = P, dfd = n - N*F -1)
        else:
            crit = chi2.ppf(q = 1-alpha, df = P)
        return wald_stat < crit # Do not reject H0
    
    # Bisection find the changing point: nb_iter = O(log_2(N))
    P_l = 1
    P_r = int(N*N/2 - N/2)
    if wald_test_oneP(P_l) == wald_test_oneP(P_r) == True:
        return xp.zeros(shape = A_N_hat.shape, dtype = A_N_hat.dtype), row_ind, col_ind
    elif wald_test_oneP(P_l) == wald_test_oneP(P_r) == False:
        return A_N_hat, [], []
    else:    
        while P_r - P_l > 1:
            P_mdl = (P_r + P_l)//2
            if wald_test_oneP(P_mdl):
                P_l = P_mdl
            else:
                P_r = P_mdl  
        for i in range(P_l):     
            A_N_hat[row_ind[i],col_ind[i]] = 0
            A_N_hat[col_ind[i],row_ind[i]] = 0  
        return A_N_hat, row_ind, col_ind
    

# Cupy complements
def triu_indices(N):
    ir = xp.arange(int(N*(N-1)*0.5))
    ic = xp.arange(int(N*(N-1)*0.5))
    nb = 0
    for i in range(N):
        for j in range(i+1, N):
            ir[nb] = i
            ic[nb] = j
            nb += 1
    return (ir,ic)
    
    
    

