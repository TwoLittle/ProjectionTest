import numpy as np 
from online_fun import *
import sys

jobid = int(sys.argv[1])
nrep = 50
cov_type = 'cs'

start_seed = 1992
np.random.seed(start_seed)
vec_seed = np.random.choice(range(9999, 9999999), size = 100000, replace = False)

n = 160
p = 400
s = 10
vecc = [0.50]
intc = int(vecc[0]*100)
vrho = [0.25, 0.5, 0.75, 0.95]
lenc = len(vecc)
lenr = len(vrho)
mu0 = np.append(np.ones(s), np.zeros(p-s))
sn = 2
#kappa = 0.4
#n1 = int(kappa*n)
batch_size = 10
df = 6

if cov_type == 'ar':
	lambds = 2**np.linspace(-2,-6,50)
if cov_type == 'cs':
	lambds = 2**np.linspace(-1,-5,50)

Y_s_all = np.zeros([4*nrep, n-sn])
Y_r_all = np.zeros([4*nrep, n-sn])
Y_s_mini = np.zeros([4*nrep, n-batch_size])
Y_r_mini = np.zeros([4*nrep, n-batch_size])

for l in range(nrep):
	for i in range(lenc):
		c = vecc[i]
		for j in range(lenr):

			r = vrho[j]
			seed_id = (jobid-1)*nrep*lenc*lenr + l*lenc*lenr + i*lenr + j
			np.random.seed(vec_seed[seed_id])

			if cov_type == 'cs':
				sigma = cov_cs(p, r)
				X = multivariate_t_rvs(c*mu0, sigma, df, n)
			if cov_type == 'ar':
				sigma = cov_AR1(p, r)
				X = multivariate_t_rvs(c*mu0, sigma, df, n)

			y_s_all = np.zeros(n-sn)
			y_r_all = np.zeros(n-sn)
			y_s_mini = np.zeros(n-batch_size)
			y_r_mini = np.zeros(n-batch_size)
	
			for k in range(sn, n):

				Xk = X[range(k),:]
				out = cqp_wl_admm_bic(Xk, lambds, 1./np.log(k), rho = 10., gamma = np.sqrt(np.log(p)/(k)))
				ws = out[0]
				y_s_all[k-sn] = np.sum(ws*X[k,:])
				wr = ridge_proj(Xk, lambd=1./np.sqrt(k))
				y_r_all[k-sn] = np.sum(wr*X[k,:])

				if k%10 == 0:
					batch_id = int(k/10)
					X_mini = X[range(batch_id*batch_size, (batch_id+1)*batch_size),:]
					y_s_mini[range((batch_id-1)*batch_size, batch_id*batch_size)] = np.dot(X_mini, ws)
					y_r_mini[range((batch_id-1)*batch_size, batch_id*batch_size)] = np.dot(X_mini, wr)

			Y_s_all[l*4+j,:] = y_s_all
			Y_r_all[l*4+j,:] = y_r_all
			Y_s_mini[l*4+j,:] = y_s_mini
			Y_r_mini[l*4+j,:] = y_r_mini

np.savetxt("yt_s_all_"+str(cov_type)+"_n"+str(n)+"_p"+str(p)+"_c"+str(intc)+"_"+str(jobid), Y_s_all, fmt = "%.6f")
np.savetxt("yt_r_all_"+str(cov_type)+"_n"+str(n)+"_p"+str(p)+"_c"+str(intc)+"_"+str(jobid), Y_r_all, fmt = "%.6f")
np.savetxt("yt_s_mini_"+str(cov_type)+"_n"+str(n)+"_p"+str(p)+"_c"+str(intc)+"_"+str(jobid), Y_s_mini, fmt = "%.6f")
np.savetxt("yt_r_mini_"+str(cov_type)+"_n"+str(n)+"_p"+str(p)+"_c"+str(intc)+"_"+str(jobid), Y_r_mini, fmt = "%.6f")