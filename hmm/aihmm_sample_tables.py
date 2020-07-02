import numpy as np
def sample_tables(trans_counts, hyperparams, beta_vec, K):   
    """
    
    Sample counts of new transitions that were unobserved previously in the 
    state indicator sequence
    
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
    
             hyperparams  - hyperparams  - dictionary to store the transition related (parameters 
                            characerizing the underlying HMM) 
                            hyperparameters: gamma0, alpha0, kappa0 and psi
    
             beta_vec     - 1xK vector containing the middle level mixing
                            paramters
    
             K            - single value denoting the truncation level for
                            the number of states in the underlying HMM
    
    Outputs: trans_counts - a dictionary containing the updated transition counts 

    CC BY-SA 3.0 Attribution-Sharealike 3.0 Yazan Qarout and Y.P. Raykov
    If you use this code in your research, please cite:
    Qarout, Y.; Raykov, Y.P.; Little, M.A. 
    Probabilistic Modelling for Unsupervised Analysis of Human Behaviour in Smart Cities. 
    Sensors 2020, 20, 784.            

    """
    
    alpha0 = hyperparams['alpha0']
    kappa0 = hyperparams['kappa0']
    
    N = trans_counts['N']
    
    M = randgen_numtable(np.vstack([alpha0*beta_vec + kappa0*np.eye(K), alpha0*beta_vec]),N)
    
    barM, sum_w = sample_barM(M, beta_vec, alpha0, kappa0)
    
    trans_counts['M'] = M
    trans_counts['barM'] = barM
    trans_counts['sum_w'] = sum_w
    
    return trans_counts

def randgen_numtable(alpha, numdata):
    """
    
    Samples a random number of tables/clusters from a Chinese restaurant process
    with concentration alpha and N data points
    
    Inputs:  alpha   - scalar denoting the concentration parameter of the CRP
             numdata - the number of elements N which characetrise the CRP
    """
    numtable = np.zeros(numdata.shape)
    for i in range(numdata.shape[0]):
        for j in range(numdata.shape[1]):
            if numdata[i,j] > 0:
                numtable[i,j] = 1 + np.sum(np.random.rand(1,numdata[i,j] - 1) < \
                np.ones([1,numdata[i,j] - 1])*alpha[i,j]/(alpha[i,j] + np.arange(1, numdata[i,j])))
            else:
                numtable[i,j] = 1
    
    numtable[numdata == 0] = 0
    
    return numtable

def sample_barM(M, beta_vec, alpha0, kappa0):
    barM = M
    sum_w = np.zeros([M.shape[1], 1])
    for j in range(M.shape[1]):
        if kappa0 > 0 and alpha0 > 0:
            p = kappa0/(alpha0*beta_vec[j] + kappa0)
        else:
            p = 0
        sum_w[j] = np.random.binomial(M[j,j],p)
        barM[j,j] = M[j,j] - sum_w[j]
        
    return barM, sum_w
