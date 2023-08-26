##---------------------------------------------------
## Define some functions
## Main function: cqp_wl_admm_bic
## output estimated optimal projection direction
## via constrained and regularized quadratic form
##---------------------------------------------------


import numpy as np
from scipy import linalg as slin
import matplotlib.pyplot as plt
from scipy.stats import t

## get the normalized z-statistic
def z_stat(y):

    n = len(y)
    stdy = np.sqrt(sum((y - np.mean(y))**2) / (n-1.))

    if sum(y**2) == 0:
        z = 0.
    elif stdy == 0.:
        z = 1000.
    else:
        z = np.mean(y)*np.sqrt(n)/stdy

    return z

## normalize a vector such that its L2 norm is 1
def vec_unit(x):
    tmp = np.sum(x**2)
    if tmp != 0:
        x = x/np.sqrt(tmp)
    return x

## define a function that generates the covariance matrix for AR(1) model
def cov_AR1(p, rho):
    mt = np.full([p,p], range(1, p+1), dtype = int)
    X  = rho**abs(mt - mt.T)
    return X

## define a function that generates the covariance matrix for compound symmetry model
def cov_cs(p, rho):
    X = np.full([p,p], rho)
    np.fill_diagonal(X, 1.)
    return X

## get normal X
def get_multivariate_normal(mu, sigma, n):
    
    p = len(mu)
    L = np.linalg.cholesky(sigma)
    tmp = np.random.randn(n, p)
    X = np.dot(tmp, L.T)
    X = X + mu
    
    return X

def multivariate_t_rvs(mu, sigma, df, n = 1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    n : int
        number of observations, return random array will be (n, len(m))
    mu : array_like
        mean of random variable, length determines dimension of random variable
    sigma : array_like
        square array of covariance matrix
    df : int or float
        degrees of freedom
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    mu = np.asarray(mu)
    p  = len(mu)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(p),sigma,(n,))
    return mu + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

## get AR1 X
def get_ar1X(n, p, rho, mu=0):
    
    X = np.zeros([n, p])
    X[:,0] = np.random.normal(0.,1.,n) 
    for k in range(1,p):
        X[:,k] = rho*X[:,k-1] + np.sqrt(1.-rho**2)*np.random.normal(0.,1.,n) 
    X = X + mu
    
    return X

## soft shresholding for min_x 1/2*(x-c)^2 + kappa*|x|
## here x and kappa are both vectors of the same length
def soft_sh(x, kappa):
    return np.maximum(0, x-kappa) - np.maximum(0, -x-kappa)

## first derivative of scad penalty: p'_lambda(|x|)
def d_scad(x, lam, a):
    absx = abs(x)
    alam = a*lam
    ind1 = absx <= lam
    ind2 = np.logical_and(absx > lam, absx <= alam)
    y    = lam*ind1 + (alam - absx)/(a-1)*ind2
    return y

## output the estimated projection direction via ridge regression
def ridge_proj(X, lambd):
    
    n, p = X.shape
    xbar = np.mean(X, axis=0)
    Xcn = (X - xbar)/np.sqrt(n)
    S = np.dot(Xcn.T, Xcn)
    d = np.diag(S)
    lamd_inv = 1./(lambd*d)
    Xcn_lamd_inv = Xcn * lamd_inv
    tmp1 = np.identity(n) + np.dot(Xcn_lamd_inv, Xcn.T)
    tmp2 = np.linalg.solve(tmp1, Xcn_lamd_inv)
    Sd_inv = np.diag(lamd_inv) - np.dot(Xcn_lamd_inv.T, tmp2)
    w = np.dot(Sd_inv, xbar)
    w = w/np.sum(w*xbar)
    
    return w

## Admm algorithm for weighted Lasso
## Reference: https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
def cqp_wl_admm(X, lambd, beta_initial = None, weights = None, rho = 1., gamma = 0., max_iter = 1000, relaxation = 1.8, 
                abstol = 1e-4, reltol = 1e-3):
    
    n, p   = X.shape
    skinny = (n >= p)
    kappa  = rho + gamma
    
    xbar = np.mean(X, axis = 0)
    Xcn  = (X - xbar)/np.sqrt(n-1)
    XcnT = Xcn.T
    rCCT = rho*np.outer(xbar, xbar)
    rCTC = rho*np.dot(xbar, xbar)
    
    if skinny:
        S = np.dot(XcnT, Xcn)
        L = np.linalg.cholesky(S + kappa * np.identity(p) + rCCT)
    else:
        #Ainv    = np.identity(p) / kappa - rCCT / (kappa * (kappa + rCTC))
        XcnAinv = Xcn/kappa - np.outer(np.dot(Xcn, xbar), xbar) * rho / (kappa * (kappa + rCTC))
        L       = np.linalg.cholesky(np.dot(XcnAinv, XcnT) + np.identity(n))
    U = L.T
    
    if beta_initial is None:
        beta_initial = np.zeros(p, dtype = float)
        
    if weights is None:
        weights = d_scad(beta_initial, lambd, 3.7)
    
    beta1 = beta_initial
    beta2 = beta_initial
    u1    = np.zeros(p)
    u2    = 0.

    ## update beta1, beta2, u
    for k in range(max_iter):
        
        q = rho * (beta2 - u1 + (1 - u2) * xbar)
        
        ## beta1 update
        if skinny:
            temp         = slin.solve_triangular(L, q, lower = True, check_finite = False)
            beta1_update = slin.solve_triangular(U, temp, lower = False, check_finite = False)
        else:
            Ainvq        = q/kappa - np.dot(xbar,q) * rho / (kappa * (kappa + rCTC)) * xbar
            temp1        = slin.solve_triangular(L, np.dot(Xcn, Ainvq), lower = True, check_finite = False)
            temp2        = slin.solve_triangular(U, temp1, lower = False, check_finite = False)
            beta1_update = Ainvq - np.dot(XcnAinv.T, temp2)
        
        ## beta2 update with relaxation
        beta1_hat    = relaxation * beta1_update + (1 - relaxation) * beta2
        beta2_update = soft_sh(beta1_hat + u1, weights/rho)
            
        ## u update
        u1_update = u1 + beta1_hat - beta2_update
        u2_update = u2 + np.dot(xbar, beta1_update) - 1.
        u_update  = np.append(u1_update, u2_update)
        
        ## residuals
        res_pri  = np.append(beta1_update - beta2_update, np.dot(xbar, beta1_update) - 1.)
        res_dual = rho*(beta2_update - beta2)
        nvareps  = np.sqrt(p)*abstol
        eps_pri  = nvareps + reltol*max(np.linalg.norm(beta1),np.linalg.norm(beta2))
        eps_dual = nvareps + reltol*np.linalg.norm(u_update)
        
        beta1 = beta1_update
        beta2 = beta2_update
        u1    = u1_update
        u2    = u2_update
        
        if np.linalg.norm(res_pri) < eps_pri and np.linalg.norm(res_dual) < eps_dual:
            break
        
    beta = beta2
    df   = np.count_nonzero(beta2)
    ite  = k + 1

    return beta, df, ite


## Admm algorithm for weighted Lasso with LLA (local linear approximation)
def cqp_wl_admm_lla(X, lambd, tau, beta_initial = None, rho = 1., gamma = 0., max_iter = 1000, relaxation = 1.8, 
                    abstol = 1e-4, reltol = 1e-3):
    n, p = X.shape
    weights1 = tau*lambd*np.ones(p)
    out1 = cqp_wl_admm(X, tau*lambd, beta_initial, weights1, rho, gamma, max_iter, relaxation, abstol, reltol)
    beta1 = out1[0]
    weights2 = d_scad(beta1, lambd, 3.7)
    out2 = cqp_wl_admm(X, lambd, beta1, weights2, rho, gamma, max_iter, relaxation, abstol, reltol)
    beta2 = out2[0]
    
    return beta1, beta2

## Main function: output the estimated optimal projection direction
## The tunning parameter lambd is selected by BIC criterion
def cqp_wl_admm_bic(X, vlambd, tau, beta_initial = None, rho = 1., gamma = 0., max_iter = 1000, relaxation = 1.8, 
                    abstol = 1e-4, reltol = 1e-3):
    
    n, p = X.shape
    nlambd = len(vlambd)
    mbeta1 = np.zeros([nlambd, p])
    mbeta2 = np.zeros([nlambd, p])
    vbic = np.zeros(nlambd)
    
    xbar = np.mean(X, axis = 0)
    Xcn  = (X - xbar) / np.sqrt(n-1)
    XcnT = Xcn.T
    #XcnC = np.dot(Xcn, xbar)
    #temp = np.linalg.solve(np.dot(Xcn, XcnT) + gamma*np.identity(n), XcnC)
    #a0   = (np.dot(xbar,xbar) - np.dot(XcnC, temp)) / gamma
    Sgamma = np.dot(XcnT, Xcn) + gamma * np.identity(p)
    a0 = np.sum(xbar*np.linalg.solve(Sgamma, xbar))
    
    
    beta1, beta2 = cqp_wl_admm_lla(X, vlambd[0], tau, beta_initial, rho, gamma, max_iter, relaxation, abstol, reltol)
    mbeta1[0] = beta1
    mbeta2[0] = beta2
    df2 = np.sum(beta2 != 0)
    sse_scale = np.sum((xbar/a0 - np.dot(Sgamma,beta2))**2)
    vbic[0] = sse_scale + df2*np.log(np.log(n))*np.log(p)/n
    
    for k in range(1,nlambd):
        beta1, beta2 = cqp_wl_admm_lla(X, vlambd[k], tau, mbeta1[k-1], rho, gamma, max_iter, relaxation, abstol, reltol)
        mbeta1[k] = beta1
        mbeta2[k] = beta2
        df2 = np.sum(beta2 != 0)
        sse_scale = np.sum((xbar/a0 - np.dot(Sgamma,beta2))**2)
        vbic[k] = sse_scale + df2*np.log(np.log(n))*np.log(p)/n
    
    index = np.argmin(vbic)
    beta = mbeta2[index]
    
    return beta, vbic