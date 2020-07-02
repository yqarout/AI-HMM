import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime as dt
from matplotlib.ticker import MaxNLocator

from aihmm_train_model import train_aihmm

# =============================================================================
# Data extraction and wrangling
# =============================================================================

# Import data and map datetime values
data = pd.read_csv('data/Dodgers.data', header = None)
labels = pd.read_csv('data/Dodgers.events', header = None, usecols = [0,1,2,3,4])
date = data[0].map(lambda x: dt.datetime.strptime(str(x), '%m/%d/%Y %H:%M').date())
time = data[0].map(lambda x: dt.datetime.strptime(str(x), '%m/%d/%Y %H:%M').time())
hour = time.map(lambda x: x.hour)
labels[0] = labels[0].map(lambda x: dt.datetime.strptime(str(x), '%m/%d/%y').date())
labels[1] = labels[1].map(lambda x: dt.datetime.strptime(str(x), '%H:%M:%S').time())

# Set AR emissions order
r=288

# Set up feature vector and normalise for analysis
Y = np.vstack([hour.values,data[1].values])
Y_Valid = Y[:,r:]
Y = (Y-np.min(Y,1).reshape(Y.shape[0],-1))/(np.max(Y,1) - np.min(Y,1)).reshape(Y.shape[0],-1) 
  
# Define input variable tau to produce differen transitions for weekdays and weekends
d = date[0]
tau = np.ones(len(data))
for i in range(len(data)):
    if date.values[i] == d - dt.timedelta(1) or date.values[i] == d:
        tau[i] = 2
        if date.values[i] != d - dt.timedelta(1) and date.values[i+1] != d:
            d = d + dt.timedelta(7)
            
# =============================================================================
# Modelling
# =============================================================================

# Dimensions and length of the input
d, T = Y.shape
m = d*r
# inverse covariance along rows of A 
P = np.linalg.inv(np.diag(0.005*np.ones(d*r)))
# variance for the mean process noise
sig0 = 1
# mean covariance for the covariance of the AR process noise
meanSigma = np.cov(Y)

# truncation level for mode transition distributions
K = 12

model = {}
model['component'] = {}
model['component']['params'] = {}
model['component']['params']['r'] = r
# Mean and covariance for A matrix
model['component']['params']['M'] = np.zeros([d,m])
# Inverse covariance along rows of A (sampled Sigma acts as covariance along columns)
model['component']['params']['P'] = P
# Mean and covariance for mean of process noise
model['component']['params']['mu0'] = np.zeros([d,1])
model['component']['params']['cholSigma0'] = np.linalg.cholesky(sig0*np.eye(d))
# Degrees of freedom and scale matrix for covariance of process noise
model['component']['params']['nu'] = d+2
model['component']['params']['nu_delta'] = (model['component']['params']['nu'] - 1)*meanSigma

# Sticky HDP-HMM parameter settings
model['HMMmodel'] = {}
model['HMMmodel']['params'] = {}
model['HMMmodel']['params']['gamma0'] = 5
model['HMMmodel']['params']['alpha0'] = 5
model['HMMmodel']['params']['kappa0'] = 10
model['HMMmodel']['params']['psi'] = 10000

# Test model
Niter = 100
state_ind_store, concat_likelihood, AR_params, trans_par = train_aihmm(Y, tau, model, K, Niter)

# Extract state indicator variables z and remove states with very few points clustered
# corrosponding to outliers for better visualisation
Z = state_ind_store[Niter-1]
short = np.array([])
for s in np.unique(Z):   
    if sum(Z==s) > 100:
        short = np.r_[short, s]  

# =============================================================================
# Single day plot
# =============================================================================

# Select day
d=3

fig3, ax3 = plt.subplots()
for s in np.unique(Z):
    ax3.scatter(np.arange(288)[Z[(288*d)-r :288+(288*d)-r] == s],\
                Y_Valid[1,(288*d)-r :288+(288*d)-r][Z[(288*d)-r :288+(288*d)-r] == s], \
                s = 8, label = 'State '+str(int(s)))
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1, fontsize = 13, markerscale = 5)

# =============================================================================
# Sequential plot
# =============================================================================
ax = {}
i = 1
d=19
dy=1

dist = 0.4
dy2 = 5
pos = 0.1+((dy2-1)*dist)

c=0
times = pd.Series(np.zeros(288*dy))
for i in np.arange(0,1440*dy,5):
    remainder = i/60
    if int(remainder)>0:            
        times[c] = dt.time(hour = int(remainder), minute = i- int(remainder)*60)
    else:
        times[c] = dt.time(minute = i)
    c += 1

weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

fig = plt.figure() 
for j in np.arange(dy2,1,-1):
    ax[j] = fig.add_axes([pos, 0.5, dist, 0.7], yticklabels=[])
    pos -= dist
    for s in short:
        ax[j].scatter(times[Z[(288*d)-r :288*dy+(288*d)-r] == s].values, \
          Y_Valid[1,(288*d)-r :288*dy+(288*d)-r][Z[(288*d)-r :288*dy+(288*d)-r] == s], \
          s = 8, label = 'State '+str(int(i)))
        i += 1
    ax[j].xaxis.set_major_locator(MaxNLocator(3))
    ax[j].set_xlim([0, 86340.0]) 
    d -= 1
    ax[j].set_title(weekdays[j-1], fontsize = 17)
    ax[j].tick_params(axis='both', which='major', labelsize=12)
    

ax[1] = fig.add_axes([0.1, 0.5, dist, 0.7])
ax[1].set_xlim([-1000, 86340.0]) 
for s in short:
    ax[1].scatter(times[Z[(288*d)-r :288*dy+(288*d)-r] == s].values, \
      Y_Valid[1,(288*d)-r :288*dy+(288*d)-r][Z[(288*d)-r :288*dy+(288*d)-r] == s], \
      s = 8, label = 'State '+str(int(i)))
    i += 1
ax[1].xaxis.set_major_locator(MaxNLocator(3))
ax[1].set_xlim([-1000, 86340.0]) 
ax[1].tick_params(axis='both', which='major', labelsize=12)

ax[1].set_title(weekdays[0], fontsize = 17)
ax[1].set_ylabel('Vehicle Count', fontsize = 17)

