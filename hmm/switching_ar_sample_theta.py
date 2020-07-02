def switching_ar_sample_theta(theta, update_stats, prior_params):
    """
    Samples the state parameters A, mu and invSigma which specify the state
    specific autoregressive processes. A containes the AR coefficients, mu is
    the AR offset parameter and invSigma is the inverse of the AR process
    noise

    Inputs: theta        -  a dictyonary that contains the state parameters: 
                            theta['invSigma'] is dxdxK matrix containing the state specific 
                            AR process noise; theta['A'] is 1xrxK matrix
                            containing the state spacific AR coefficients; 
                            theta['mu'] is dxK matrix containing the state
                            specific AR offset.
    
            update_stats -  a dictionary containing sufficient statistics
                            required for the efficient parameter updates.
                            Using X to denote the reformatted lag matrix form of the
                            data and Y the input raw form:
                            -> update_stats['card'] is 1xK vector obtained by a row sum of the transition counts  
                            -> update_stats['XX'] is rxrxK matrix obtained by the sum of products X*X'  
                            -> update_stats['YX'] is 1xrxK matrix obtained by the sum of products Y*X' 
                            -> update_stats['YY'] is 1x1xK vector obtained by the sum of products Y*Y'  
                            -> update_stats['Y'] is 1xK matrix obtained by the sum over Y  
                            -> update_stats['X'] is rxK matrix obtained by the sum over X 
    
            prior_params -  dictionary containing various model prior parameters.
                            In this function we use the fields r, M, P, mu0,cholSigma0, nu, nu_delta 
                            -> M          - mean for the AR coefficients A 
                            -> P          - inverse covariance along the rows
                                            of A (used to sample the covariance between the coefficients)                         
                            -> mu0        - prior mean for the mean of the AR process noise 
                            -> cholSigma0 - prior covariance for the mean of the AR process noise
                            -> nu         - degress of freedom for the covariance of the AR process noise
                            -> nu_delta   - scale matrix for the covariance
                                            of the AR process noise 
    
    Outputs: theta       -  updated theta dictionary

    Copyright 2019 Yazan Qarout
    If you use this code in your research, please cite:
    Qarout, Y.; Raykov, Y.P.; Little, M.A. 
    Probabilistic Modelling for Unsupervised Analysis of Human Behaviour in Smart Cities. 
    Sensors 2020, 20, 784.             

    """
    
    nu = prior_params['nu']
    nu_delta = prior_params['nu_delta']
    store_card = update_stats['card']
    
    K = len(store_card)
    
    invSigma = theta['invSigma']
    A = theta['A']
    mu = theta['mu']
    
    XX = update_stats['XX']
    YX = update_stats['YX']
    YY = update_stats['YY']
    sumY = update_stats['sumY']
    sumX = update_stats['sumX']
    
    P = prior_params['P']
    M = prior_params['M']
    MP = np.dot(M,P)
    MKP = np.dot(MP,M.T)
    
    numIter = 10
    mu0 = prior_params['mu0']
    cholSigma0 = prior_params['cholSigma0']
    Lambda0 = np.linalg.inv(np.dot(prior_params['cholSigma0'].T, prior_params['cholSigma0']))
    theta0 = np.dot(Lambda0, mu0)

    dimu = nu_delta.shape[0]
    for k in range(K):
        if store_card[k] > 0:
            for n in range(numIter):
                Sxx = XX[:,:,k] + P

                Syx = YX[:,:,k] + MP - np.dot(mu[:,k].reshape(-1,1),sumX[:,k].reshape(1,-1))
                Syy = YY[:,:,k] + MKP - 2*np.dot(mu[:,k].reshape(-1,1),sumY[:,k].reshape(1,-1)) \
                    + store_card[k]*np.dot(mu[:,k].reshape(-1,1), mu[:,k].reshape(1,-1))
                
                SyxSxxInv = np.dot(Syx,np.linalg.inv(Sxx))
                Sygx = Syy - np.dot(SyxSxxInv,Syx.T)
                Sygx = (Sygx + Sygx.T)/2

                sqrtSigma, sqrtinvSigma = randgen_invwishart(Sygx + nu_delta, nu + store_card[k])
                
                invSigma[:,:,k] = np.dot(sqrtinvSigma.T, sqrtinvSigma)
                
                cholinvSxx = np.linalg.cholesky(np.linalg.inv(Sxx))
                A[:,:,k] = randgen_matrixNormal(SyxSxxInv, sqrtSigma, cholinvSxx)
                
                Sigma_n = np.linalg.inv(Lambda0 + store_card[k]*invSigma[:,:,k])
                mu_n = np.dot(Sigma_n, theta0 + np.dot(invSigma[:,:,k],sumY[:,k] - np.dot(A[:,:,k], sumX[:,k])).reshape(theta0.shape))

                mu[:,k] = (mu_n + np.dot(np.linalg.cholesky(Sigma_n).T, np.random.randn(dimu,1))).reshape(mu[:,k].shape)
                
        else:
            sqrtSigma, sqrtinvSigma = randgen_invwishart(nu_delta, nu)
            invSigma[:,:,k] = np.dot(sqrtinvSigma.T, sqrtinvSigma)
            cholinvK = np.linalg.cholesky(np.linalg.inv(P))
            A[:,:,k] = randgen_matrixNormal(M, sqrtSigma, cholinvK)
            mu[:,k] = (mu0 + np.dot(cholSigma0.T, np.random.rand(dimu,1))).reshape(mu[:,k].shape)                
            
    theta['invSigma'] = invSigma
    theta['A'] = A
    theta['mu'] = mu
    
    return theta

def randgen_invwishart(sigma, df):
    """
    Generate samples from inverse Wishart distribution with scale matrix sigma 
    and degrees of freedom df
    
    Inputs:  sigma   - nxn square scale matrix parameter for the inverse Wishart
                      distribution
             df      - a scalar which must satisfies df > n-1, denoting the
                      degrees of the inverse Wishart distribution
                      
    Outputs: sqrtx    - nxn elementwise square root matrix of the random invWishart 
             sqrtinvx - nxn elementwise square root of the inverse of sqrtx (i.e. Wishart random sample)
    """
    
    n = sigma.shape[0]
    if df < n:
        raise ValueError('randwish: Bad df, Degrees of freedom must be no smaller than dimesion of sigma')
    
    di = np.linalg.inv(np.linalg.cholesky(sigma).T)
    
    cholX = np.triu(np.random.randn(n,n))*np.sqrt(0.5)
    a = (df/2) - np.arange(n)*0.5
    gam = np.sqrt(sp.random.gamma(a)) 
    I = np.eye(cholX.shape[0], dtype = 'bool')
    cholX[I] = gam
    
    sqrtinvx =  np.sqrt(2)*np.dot(cholX,di)
    sqrtx = np.linalg.inv(sqrtinvx).T
    
    return sqrtx, sqrtinvx
    
def randgen_matrixNormal(M, sqrtV, sqrtinvK):
    """
    Generates samples from a matrix Gaussian distribution with parameters: mean M, column scale sqrtV and row scale sqrtinvK 
    """
    mu = M.ravel()
    sqrtsigma = np.kron(sqrtinvK, sqrtV)
    A = (mu + np.dot(sqrtsigma.T, np.random.randn(len(mu),1)).ravel()).reshape(M.shape)
    
    return A