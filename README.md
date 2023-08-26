# Projection Test
Projection Test for High-dimensional Mean Vectors


## Functions

online_fun.py: 
This file contains all functions needed to estimate the optimal projection direction. The main function is cqp_wl_admm_bic, which outputs estimated optimal projection direction via constrained and regularized quadratic programming. The tuning parameter is selected by BIC criterion.


## Simulation settings

#### Simulation results for normal distribution cases:
* arn4p16c0.py: results for autoregression with n=400, p=1600, c=0.0
* arn4p16c5.py: results for autoregression with n=400, p=1600, c=0.5
* arn4p16c10.py: results for autoregression with n=400, p=1600, c=1.0
* csn4p16c0.py: results for compound symmetry with n=400, p=1600, c=0.0
* csn4p16c5.py: results for compound symmetry with n=400, p=1600, c=0.5
* csn4p16c10.py: results for compound symmetry with n=400, p=1600, c=1.0

#### Simulation results for t-distribution cases:
* t6_arn4p16c0.py: results for autoregression with n=400, p=1600, c=0.0
* t6_arn4p16c5.py: results for autoregression with n=400, p=1600, c=0.5
* t6_arn4p16c10.py: results for autoregression with n=400, p=1600, c=1.0
* t6_csn4p16c0.py: results for compound symmetry with n=400, p=1600, c=0.0
* t6_csn4p16c5.py: results for compound symmetry with n=400, p=1600, c=0.5
* t6_csn4p16c10.py: results for compound symmetry with n=400, p=1600, c=1.0

## Shell

multiPy_mc.sh: A shell script which allows you to submit the same job multiple times with different seeds

## How to use
In your terminal, do "sh multiPy_mc.sh t6_arn4p16c10 100" where 100 can be replaced by any integer, which means how many times you want to repeat this job (with different seeds)


## Reference

Liu, W., Yu, X., Zhong, W., & Li, R. (2022). Projection test for mean vector in high dimensions. Journal of the American Statistical Association, 1-13.

Link: https://www.tandfonline.com/doi/abs/10.1080/01621459.2022.2142592
