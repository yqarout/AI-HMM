import numpy as np
import makeDesignMatrix from utils
def aihmm_utils_initialize(prior_params, Y, K, tau):
    """
     
    Initialize structures storing the model parameters and efficient
    representation of the data (extended lag form) which leads to faster updates in Python
    implementation.
   
    Inputs:  prior_params - dictionary containing various model prior parameters.
                            In this function we use the fields r, M, P, mu0,cholSigma0, nu, nu_delta 
                            -> r          - maximum order of the the state specific AR models;
                            -> M          - mean for the AR coefficients A (set to 0 for the demo)
   
             Y            - matrix that contains the raw input data dxT
   
             K            - single value denoting the truncation level for
                            the number of states in the underlying infinite HMM
      
             tau          - 1xT array containing the values of the input variable tau
   
    Outputs: theta        - a dictyonary that contains the state parameters: 
                            theta['invSigma'] is dxdxK matrix containing the state specific 
                            AR process noise; theta['A'] is 1xrxK matrix
                            containing the state spacific AR coefficients; 
                            theta['mu'] is dxK matrix containing the state
                            specific AR offset.
   
             update_stats - a dictionary containing sufficient statistics
                            required for the efficient parameter updates.
                            Using X to denote the reformatted lag matrix form of the
                            data and Y the input raw form:
                            -> update_stats['card'] is 1xK vector obtained by a row sum of the transition counts  
                            -> update_stats['XX'] is rxrxK matrix obtained by the sum of products X*X'  
                            -> update_stats['YX'] is 1xrxK matrix obtained by the sum of products Y*X' 
                            -> update_stats['YY'] is 1x1xK vector obtained by the sum of products Y*Y'  
                            -> update_stats['Y'] is 1xK matrix obtained by the sum over Y  
                            -> update_stats['X'] is rxK matrix obtained by the sum over X 
   
             trans_counts - dictionary containing the transition counts and
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
   
             data_struct  - dictionary that contains the raw input data and
                            reformatted form of the data which speed ups the
                            updates: data_struct['obs'] contains the input dxT
                            matrix; data_struct['X'] contains the reformatted lag matrix;
                            data_struct['tau'] 1xT array containing the values of the input variable tau
    
    CC BY-SA 3.0 Attribution-Sharealike 3.0 Yazan Qarout and Y.P. Raykov
    If you use this code in your research, please cite:
    Qarout, Y.; Raykov, Y.P.; Little, M.A. 
    Probabilistic Modelling for Unsupervised Analysis of Human Behaviour in Smart Cities. 
    Sensors 2020, 20, 784.             
        
    """
     
    dimu, dimX = prior_params['M'].shape
    
    theta = {'invSigma': np.zeros([dimu,dimu,K]), 'A': np.zeros([dimu,dimX,K]), 'mu': np.zeros([dimu,K])}
    
    update_stats = {'card': np.zeros(K), 'XX': np.zeros([dimX,dimX,K]), 'YX': np.zeros([dimu,dimX,K]), \
                    'YY': np.zeros([dimu,dimu,K]), 'sumY': np.zeros([dimu,K]), 'sumX': np.zeros([dimX,K])}
    
    X, valid = makeDesignMatrix(Y, prior_params['r'])
    data_struct = {}
    data_struct['obs'] = Y[:, valid]
    data_struct['X'] = X[:, valid]
    data_struct['tau'] = tau[valid]
    
    trans_counts = {}
    for l in np.unique(tau):
        trans_counts[l] = {}
        trans_counts[l]['N'] = np.zeros([K+1,K])
        trans_counts[l]['M'] = np.zeros([K+1,K])
        trans_counts[l]['barM'] = np.zeros([K+1,K])
        trans_counts[l]['sum_w'] = np.zeros([1,K])

    return theta, update_stats, trans_counts, data_struct
