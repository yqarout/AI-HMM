import numpy as np
from utils import randgen_dirichlet

def sample_trans_par(trans_counts, hyperparams, lamb):
    """
    
    Samples transition matrix and middle level DP mixing weights for the underlying 
    HMM.

    Inputs:  trans_counts - dictionary containing the transition counts and
                            a count of the different transitions pointing to
                            each state accross the different realisations of tau: 
                            trans_counts['N'] is (K+1)xK matrix
                            counting the number of transitions that have
                            occured from state i to state j, the extra row occounts 
                            for the support of creating a new transition;
                            trans_counts['barM'] is (K+1)xK matrix counting the
                            number of different states pointing to each
                            state, i.e. trans.counts['barM'][i,j] counts the number of
                            different transitions. 

             hyperparams  - dictionary to store the transition related (parameters 
                            characerizing the underlying HMM) 
                            hyperparameters: gamma0, alpha0, kappa0 and psi
                            
             lamb         - the upper level mixing parameter lambda 
                             
    Outputs: trans_par    - structure containing the sampled transition
                            parameters: pi_z contains the (K+1)xK transition
                            matrix; pi_init contains the 1xK initial transition
                            weights; beta_vec is 1xK vector containing the top level mixing
                            paramters

    CC BY-SA 3.0 Attribution-Sharealike 3.0 Yazan Qarout and Y.P. Raykov
    If you use this code in your research, please cite:
    Qarout, Y.; Raykov, Y.P.; Little, M.A. 
    Probabilistic Modelling for Unsupervised Analysis of Human Behaviour in Smart Cities. 
    Sensors 2020, 20, 784.           
    
    """
    
    K = trans_counts['N'].shape[1]
    
    alpha0 = hyperparams['alpha0']
    kappa0 = hyperparams['kappa0']

    N = trans_counts['N']
    barM = trans_counts['barM']
    gamma0 = hyperparams['gamma0']
    beta_vec = randgen_dirichlet(np.sum(barM,0) + gamma0*lamb)
    
    pi_z = np.zeros([K,K])
    for j in range(K):
        kappa_vec = np.zeros(K)
        kappa_vec[j] = kappa0
        pi_z[j,:] = randgen_dirichlet(alpha0*beta_vec + kappa_vec + N[j,:])

    pi_init = randgen_dirichlet(alpha0*beta_vec + N[K,:])
    trans_par = {}
    trans_par['pi_z'] = pi_z
    trans_par['pi_init'] = pi_init
    trans_par['beta_vec'] = beta_vec
    
    return trans_par


