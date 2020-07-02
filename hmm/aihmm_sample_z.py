import numpy as np
from aihmm_likelihood import compute_likelihood

def sample_z(data_struct, trans_par, theta, valid_tau):
    """
    
    Samples the state indicator variables z which denotes the state that 
    is associated with each observation.
    
    Inputs:  data_struct  - dictionary that contains the raw input data and
                            reformatted form of the data which speed ups the
                            updates: data_struct['obs'] contains the input dxT
                            matrix; data_struct['X'] contains the reformatted lag matrix;
                            data_struct['tau'] 1xT array containing the values of the input variable tau
     
             trans_par    - structure containing the sampled transition
                            parameters: pi_z contains the (K+1)xK transition
                            matrix; pi_init contains the 1xK initial transition
                            weights; beta_vec is 1xK vector containing the top level mixing
                            paramters
                            
             theta        - a dictyonary that contains the state parameters: 
                            theta['invSigma'] is dxdxK matrix containing the state specific 
                            AR process noise; theta['A'] is 1xrxK matrix
                            containing the state spacific AR coefficients; 
                            theta['mu'] is dxK matrix containing the state
                            specific AR offset.
                        
             valid_tau    - array containing the values of the input variable tau
    
    Outputs: state_ind     - dictionary with a single field z which contains 1xT
                             vector with the associate state idicator values for each point 
                             
             ind_struct    - dictionary with the field obsIndzs which all has
                             sub-fields tot and inds: 
                             tot is denotes the total number of points belonging to state k; 
                             inds is a KxT matrix which stores the indices of points 
                             assigned to state k. 
                             
             trans_counts  - dictionary containing the transition counts and
                             a count of the different transitions pointing to
                             each state accross the different realisations of tau 
                             accross the different realisations of tau: 
                             trans_counts['N'] is (K+1)xK matrix
                             counting the number of transitions that have
                             occured from state i to state j, the extra row occounts 
                             for the support of creating a new transition;
                             trans_counts['barM'] is (K+1)xK matrix counting the
                             number of different states pointing to each
                             state, i.e. trans.counts['barM'][i,j] counts the number of
                             different transitions. 
    
             likelihood    - a single value reflecting the log likelihood of
                             the data given the sampled state indicators 

    CC BY-SA 3.0 Attribution-Sharealike 3.0 Yazan Qarout and Y.P. Raykov
    If you use this code in your research, please cite:
    Qarout, Y.; Raykov, Y.P.; Little, M.A. 
    Probabilistic Modelling for Unsupervised Analysis of Human Behaviour in Smart Cities. 
    Sensors 2020, 20, 784.  

    """
    
    pi_z = {}
    pi_init = {}
    for tau in trans_par:
        if tau != 'lambda':
            pi_z[tau] = trans_par[tau]['pi_z']
            pi_init[tau] = trans_par[tau]['pi_init']
            
    K = pi_z[tau].shape[1]
    N = {}
    for tau in np.unique(valid_tau):
        N[tau] = np.zeros([K+1,K], dtype = 'int')
    
    T = data_struct['obs'].shape[1]
    ind_struct = {}
    ind_struct['obsIndzs']  = {'inds': np.zeros([K,T], dtype = 'int'), 'tot': np.zeros([K], dtype = 'int')}
    state_ind = {'z': np.zeros(T)}
    
    z = np.zeros(T, dtype = 'int')
    
    likelihood = compute_likelihood(data_struct, theta, K)
    
    bwds_msg, partial_marg = backwards_message_vec(likelihood, T, K, pi_z, valid_tau)
    
    totSeq = np.zeros([K], dtype = 'int')
    indSeq = np.zeros([T,K,1], dtype = 'bool')
    likelihood_contribution = np.zeros([T,1])
    for t in range(T):
        if t == 0:
            Pz = np.multiply(pi_init[valid_tau[t]].T, partial_marg[:,t])
        else:
            Pz = np.multiply(pi_z[valid_tau[t]][z[t-1],:].T, partial_marg[:,t])
            
        Pz = np.cumsum(Pz)

        z[t] = 0 + sum(Pz[-1]*np.random.rand(1) > Pz)
        likelihood_contribution[t] = Pz[z[t]]
        
        if t > 0:
            N[valid_tau[t]][z[t-1],z[t]] = N[valid_tau[t]][z[t-1],z[t]] + 1
        else:
            N[valid_tau[t]][K,z[t]] = N[valid_tau[t]][K,z[t]] + 1
            
        totSeq[z[t]] = totSeq[z[t]] + 1
        indSeq[t,z[t],0] = 1
        
    likelihood = sum(likelihood_contribution)
    state_ind['z'] = z
    
    ind_struct['obsIndzs']['tot'] = totSeq
    ind_struct['obsIndzs']['inds'] = indSeq[:,:,0].T
         
    trans_counts = {}
    for tau in np.unique(valid_tau):
        trans_counts[tau] = {}
        trans_counts[tau]['N'] = N[tau]
    
    return state_ind, ind_struct, trans_counts, likelihood 

def backwards_message_vec(likelihood, T, K, pi_z, valid_tau):    
    bwds_msg = np.ones([K,T])
    partial_marg = np.zeros([K,T])
        
    for tt in np.arange(T-2,-1,-1):
        partial_marg[:,tt+1] = np.multiply(likelihood[:,tt+1], bwds_msg[:,tt+1])
        bwds_msg[:,tt] = np.dot(pi_z[valid_tau[tt+1]], partial_marg[:,tt+1])
        bwds_msg[:,tt] = bwds_msg[:,tt] / sum(bwds_msg[:,tt])
        
    partial_marg[:,0] = np.multiply(likelihood[:,0], bwds_msg[:,0])
    
    return bwds_msg, partial_marg    

