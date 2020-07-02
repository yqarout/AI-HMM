import numpy as np
def update_stats(data_struct, ind_struct, N):
    """
    
    Update the sufficient statistics for each state to speed up the sampling
    for the state parametes theta.
    
    Input:     data_struct  - dictionary that contains the raw input data and
                              reformatted form of the data which speed ups the
                              updates: data_struct['obs'] contains the input dxT
                              matrix; data_struct['X'] contains the reformatted lag matrix;
                              data_struct['tau'] 1xT array containing the values of the input variable tau
                              
               ind_struct   - dictionary with the field obsIndzs which all has
                              sub-fields tot and inds: 
                              tot is denotes the total number of points belonging to state k; 
                              inds is a KxT matrix which stores the indices of points 
                              assigned to state k.
                              
               N            - a (K+1)xK matrix counting the number of transitions that have
                              occured from state i to state j for all the 
                              trastition counts accross all the realisations of tau.
    
    Output:   update_stats  - update_stats - a dictionary containing sufficient statistics
                              required for the efficient parameter updates.
                              Using X to denote the reformatted lag matrix form of the
                              data and Y the input raw form:
                              -> update_stats['card'] is 1xK vector obtained by a row sum of the transition counts  
                              -> update_stats['XX'] is rxrxK matrix obtained by the sum of products X*X'  
                              -> update_stats['YX'] is 1xrxK matrix obtained by the sum of products Y*X' 
                              -> update_stats['YY'] is 1x1xK vector obtained by the sum of products Y*Y'  
                              -> update_stats['Y'] is 1xK matrix obtained by the sum over Y  
                              -> update_stats['X'] is rxK matrix obtained by the sum over X
                              
    CC BY-SA 3.0 Attribution-Sharealike 3.0 Yazan Qarout and Y.P. Raykov
    If you use this code in your research, please cite:
    Qarout, Y.; Raykov, Y.P.; Little, M.A. 
    Probabilistic Modelling for Unsupervised Analysis of Human Behaviour in Smart Cities. 
    Sensors 2020, 20, 784.                              
        
    """   
    
    K = N.shape[1]
    unique_z = np.where(np.sum(N,0) != 0)[0]
    
    dimu = data_struct['obs'].shape[0]
    dimX = data_struct['X'].shape[0]
    
    XX = np.zeros([dimX,dimX,K])
    YX = np.zeros([dimu,dimX,K])
    YY = np.zeros([dimu,dimu,K])
    sumY = np.zeros([dimu,K])
    sumX = np.zeros([dimX,K])
    
    u = data_struct['obs']
    X = data_struct['X']
    
    for k in unique_z:
        obsInd = ind_struct['obsIndzs']['inds'][k,:]
        XX[:,:,k] = XX[:,:,k] + np.dot(X[:,obsInd],X[:,obsInd].T)
        YX[:,:,k] = YX[:,:,k] + np.dot(u[:,obsInd],X[:,obsInd].T)
        YY[:,:,k] = YY[:,:,k] + np.dot(u[:,obsInd],u[:,obsInd].T)
        sumY[:,k] = sumY[:,k] + np.sum(u[:,obsInd],1)
        sumX[:,k] = sumX[:,k] + np.sum(X[:,obsInd],1)
    
    update_stats = {}
    update_stats['card'] = np.sum(N, 0)
    update_stats['XX'] = XX
    update_stats['YX'] = YX
    update_stats['YY'] = YY
    update_stats['sumY'] = sumY
    update_stats['sumX'] = sumX
    
    return update_stats
