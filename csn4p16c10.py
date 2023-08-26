import numpy as np 
from online_fun import *
import sys

jobid = int(sys.argv[1])
nrep = 10
cov_type = 'cs'

start_seed = 1024
np.random.seed(start_seed)
vec_seed = np.random.choice(range(9999, 9999999), size = 100000, replace = False)

n = 40
p = 1600
s = 10
vecc = [1.0]
intc = int(vecc[0]*10)
vrho = [0.25, 0.5, 0.75, 0.95]
lenc = len(vecc)
lenr = len(vrho)
mu0 = np.append(np.ones(s), np.zeros(p-s))
sn = 2
kappa = 0.4
n1 = int(kappa*n)

if cov_type == 'ar':
	lambds = 2**np.linspace(-2,-6,50)
if cov_type == 'cs':
	lambds = 2**np.linspace(-1,-5,50)

Y = np.zeros([4*nrep, n-sn])

for l in range(nrep):
	for i in range(lenc):
		c = vecc[i]
		for j in range(lenr):

			r = vrho[j]
			seed_id = (jobid-1)*nrep*lenc*lenr + l*lenc*lenr + i*lenr + j
			np.random.seed(vec_seed[seed_id])

			if cov_type == 'cs':
				sigma = cov_cs(p, r)
				X = get_multivariate_normal(c*mu0, sigma, n)
			if cov_type == 'ar':
				X = get_ar1X(n, p, r, c*mu0)

			y_s_on = np.zeros(n-sn)
	
			for k in range(sn, n):

				Xk = X[range(k),:]
				out = cqp_wl_admm_bic(Xk, lambds, 1./np.log(k), rho = 10., gamma = np.sqrt(np.log(p)/(k)))
				ws = out[0]
				y_s_on[k-sn] = np.sum(ws*X[k,:])

			Y[l*4+j,:] = y_s_on

np.savetxt("y_"+str(cov_type)+"_n"+str(n)+"_p"+str(p)+"_c"+str(intc)+"_"+str(jobid), Y, fmt = "%.6f")