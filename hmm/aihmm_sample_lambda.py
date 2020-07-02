import numpy as np
from utils import randgen_dirichlet

def sample_lambda(hyperparams, M):
    """   
    Sample the upper level mixing parameter lambda
    
    Inputs:  hyperparams  -  dictionary to store the transition related (parameters 
                             characerizing the underlying HMM) 
                             hyperparameters: gamma0, alpha0, kappa0 and psi
                             
             M            -  counts of new transitions that were unobserved previously in the 
                             state indicator sequence and sampled from the upper level DP
   
    Outputs: lamb         - the resampled update of the upper level mixing parameter lambda
    
    CC BY-SA 3.0 Attribution-Sharealike 3.0 Yazan Qarout and Y.P. Raykov
    If you use this code in your research, please cite:
    Qarout, Y.; Raykov, Y.P.; Little, M.A. 
    Probabilistic Modelling for Unsupervised Analysis of Human Behaviour in Smart Cities. 
    Sensors 2020, 20, 784.

    """
    
    psi = hyperparams['psi']
    L = M.shape[1]
    lamb = randgen_dirichlet(np.sum(M,0) + psi/L)
    return lamb





