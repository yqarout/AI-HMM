import numpy as np
import matplotlib.pyplot as plt
from aihmm_initialize import initialize
from aihmm_sample_lambda import sample_lambda
from aihmm_sample_trans_par import sample_trans_par
from aihmm_sample_theta import sample_theta
from aihmm_sample_z import sample_z
from aihmm_sample_tables import sample_tables
from aihmm_update_stats import update_stats_f

def train_aihmm(Y, tau, model, K, Niter):
    unique_tau = np.unique(tau)
    prior_HMM_params = model['HMMmodel']['params']
    prior_params  = model['component']['params']  
    theta, update_stats, trans_counts, data_struct = initialize(prior_params, Y, K, tau)
     
    trans_par = {}
    
    trans_par['lambda'] = sample_lambda(prior_HMM_params, np.zeros([K+1,K]))
    
    for l in unique_tau:
        trans_par[l] = sample_trans_par(trans_counts[l], prior_HMM_params, trans_par['lambda'])
    
    theta = sample_theta(theta, update_stats, prior_params)
    
    valid_tau = data_struct['tau']
    
    concat_likelihood = np.array([])
    ind_struct = {}
    state_ind_store = {}
    AR_params = {}
    
    for n in range(Niter):
        concat_N = np.zeros([K+1, K], dtype = 'int')
        sum_barM = np.zeros([K+1, K], dtype = 'float64')
        
        state_ind, ind_struct, trans_counts, likelihood = sample_z(data_struct, trans_par, theta, valid_tau)
        for l in unique_tau:
            concat_N += trans_counts[l]['N']   
            trans_counts[l] = sample_tables(trans_counts[l], prior_HMM_params, trans_par[l]['beta_vec'], K)
            
            sum_barM += trans_counts[l]['barM']
            
        trans_par['lambda'] = sample_lambda(prior_HMM_params, sum_barM)
        
        for l in unique_tau:
            trans_par[l] = sample_trans_par(trans_counts[l], prior_HMM_params, trans_par['lambda'])

        update_stats = update_stats(data_struct, ind_struct, concat_N)
        theta = sample_theta(theta, update_stats, prior_params)
        state_ind_store[n] = state_ind['z']
        AR_params[n] = theta
        concat_likelihood = np.r_[concat_likelihood, likelihood]
        
        print('Iteration number: '+str(n))
        if not n %10:
            plt.figure(1)
            plt.plot(state_ind_store[n])
            plt.title('State Indicator Z')
            plt.show()
            
    return state_ind_store, concat_likelihood, AR_params, trans_par   
