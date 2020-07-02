import numpy as np
import scipy as sp

def randgen_dirichlet(a):
    """   
    Generate samples from a Dirichlet distribution
    
    Input:  a        - 1xK vector which specifies the concentration parameters for the Dirichlet distribution
    
    OutPut: x/sum(x) - 1xK vector of Dirichlet samples
    
    """    
    x = np.random.gamma(a)
    return x/sum(x)

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
    """
    
    Updates component level transition counts M (KxK matrix) which we marginalize 
    per row to obtain the sufficient statistics for update on beta_vec mixing parameter
    """
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

def makeDesignMatrix(Y, order):
    """
    Create a lag matrix which stores in each row lagged input observation Y with lag ranging from 1 to input order   
    """
    d, T = Y.shape
    
    X = np.zeros([order*d, T])
    
    for lag in range(order):
        ii = (d*lag)
        X[ii:ii+d,lag+1:] = Y[:,:T-min(lag+1,T)]
        
    valid = np.ones(T, dtype = 'bool')
    valid[:order] = 0
    
    return X, valid

def compute_ar_likelihood(data_struct, theta, K):
    """
    Computes the likelihood of input data, stored in data_struct, given state parameters theta and state number K
    """
    invSigma = theta['invSigma']
    A = theta['A']
    X = data_struct['X']
    
    dimu, T = data_struct['obs'].shape
    
    log_likelihood = np.zeros([K,T]);
    
    if 'mu' in theta:
        mu = theta['mu']
        for k in range(K):
            cholinvSigma = np.linalg.cholesky(invSigma[:,:,k])
            dcholinvSigma = np.diag(cholinvSigma)
            
            u = np.dot(cholinvSigma,(data_struct['obs'] - np.dot(A[:,:,k],X) - mu[:,k].reshape(dimu,-1)))
            
            log_likelihood[k,:] = -0.5*np.sum(u**2, 0) + sum(np.log(dcholinvSigma))
            
        else:
            cholinvSigma = np.linalg.cholesky(invSigma[:,:,k])
            dcholinvSigma = np.diag(cholinvSigma)
            
            u = np.dot(cholinvSigma,(data_struct['obs'] - np.dot(A[:,:,k],X)))
            
            log_likelihood[k,:] = -0.5*np.sum(u**2, 0) + sum(np.log(dcholinvSigma))
        
        normalizer = np.max(log_likelihood, 0)
        log_likelihood = log_likelihood - normalizer
        log_likelihood = np.exp(log_likelihood)
        
    return log_likelihood

