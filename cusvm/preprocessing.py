# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:27:18 2020

@author: sopmathieu

This script contains different functions that could be used to preprocess the 
data and isolate an interesting component for the monitoring. 
They also allow us to automatically choose a subset of stable series from the 
panel and to standardise the data with time-varying mean and variance. 

"""

import numpy as np 
from scipy.stats import iqr
from sklearn.cluster import MeanShift, estimate_bandwidth,KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from kneed import KneeLocator
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)


def rescaling(data, period_rescaling):
    """
    Rescales the data wrt the median of the panel.
    
    This function rescales the observations on the median of the panel using 
    piece-wise constant scaling-factors. These factors are computed using a 
    simple linear regression of the observations on the median without 
    intercept.
    
    Parameters
    ----------
    data : 2D-array 
        A matrix of observations to be rescaled (rows: time, columns: series).
    period_rescaling : float>0
        Length of the period where the scaling-factors are assumed to be 
        constant, expressed in number of observations.
        
    Returns: 
    -------
    obs_rescaled : 2D-array
        A matrix with the rescaled observations
        
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    assert period_rescaling > 0, "Period must be strictly positive"
    (n_obs, n_series) = data.shape
    step = int(period_rescaling)
    n = int(np.ceil(n_obs/step))
    
    med = np.nanmedian(data, axis=1)#create warnings: sometimes all values are Nans

    #### linear regression
    slopeOLSY = np.ones((n, n_series)) 
    for j in range(n_series):
        c = 0
        for i in range(0, n):
            X = data[c:c+step,j]
            Y = med[c:c+step]
            ind = np.where(~np.isnan(X))
            Y = Y[ind].reshape(-1,1) 
            X = X[ind].reshape(-1,1)
            c += step
            if len(Y > 0) > 2 and len(X > 0) > 2:
                reg = LinearRegression(fit_intercept=False).fit(X, Y)
                slopeOLSY[i,j] = reg.coef_ #slope
        
    
    ### effective rescaling 
    obs_rescaled = np.zeros((n_obs, n_series)) 
    for j in range(n_series):
        c = 0
        for i in range(0, n):
            if slopeOLSY[i,j] > 0 and not np.isnan(slopeOLSY[i,j]):
                obs_rescaled[c:c+step,j] = data[c:c+step,j] * slopeOLSY[i,j]
            c += step
    
    return obs_rescaled, slopeOLSY


def median(data):
    """ 
    Computes the median of a panel of obs. along the time.
    
    Parameters
    ----------
    data : 2D-array 
        A matrix of observations (rows: time, columns: series).
        
    Returns
    ------
    Mt : 1D-array
        Daily median (reference) of the panel.
        
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (nobs, n_series) = data.shape
    Mt = np.nanmedian(data, axis=1)
    
    return Mt

def remove_signal(data, model='multiplicative', ref=None, n_add=None):
    """ 
    This function removes the common signal of a panel of series. 
    
    Depending on the model, the function subtracts the estimated common 
    signal from the obs (if model='additive') or it divides 
    the obs. by the estimated signal (if model='multiplicative').
    
    Parameters
    ---------
    data : 2D-array 
        A matrix of observations (rows: time, columns: series).
    model : string, optional
        A variable that designates the type of model. 
        It takes only two different values: 'multiplicative' or 'additive'.
        The default is 'multiplicative'.
    ref : 1D-array, optional
        The reference of the panel. When set to 'None', the median of the panel 
        is used. The default is None. 
    n_add : float, optional
        Number added to the denominator when we divide the obs by the 
        estimated signal (otherwise we obtain too high values around minima). 
        When None, it is equal to 0.02*max(estimated signal).
        The default is None. 
         
    Returns
    ------
    ratio : 2D-array
        The data without the common signal.
        
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    assert model in ['additive','multiplicative'], "Undefined model"
    (n_obs, n_series) = data.shape
    if ref is None: 
        Mt = np.nanmedian(data, axis=1)
    else :
        Mt = ref
    if n_add is None:
        max_val = max(Mt[~np.isnan(Mt)])
        n_add = 0.02*max_val
        
    if model == 'multiplicative':
        ratio = np.zeros((n_obs, n_series))
        ratio[:] = np.nan
        for i in range(n_obs):
            for j in range(n_series):
                if not np.isnan(Mt[i]) and Mt[i] > 0:
                    ratio[i,j] = data[i,j]/(Mt[i]+n_add)
                if Mt[i] == 0 and data[i,j] == 0 : 
                    ratio[i,j] = 0
                    
    if model =='additive':
        ratio = np.zeros((n_obs, n_series))
        ratio[:] = np.nan
        for i in range(n_obs):
            for j in range(n_series):
                if not np.isnan(Mt[i]) :
                    ratio[i,j] = data[i,j] - Mt[i] 
        
    return ratio


#======================================================================
#======================================================================

def level_removal(x, wdw, center=True):
    """
    Removes the intrinsic levels of a panel of series.
    
    This function applies a smoothing process in time by a moving-average
    (MA) filter on each individual series. Then, the smoothed series are
    substracted from the initial series to remove the levels of the 
    processes.
    
    Parameters
    ---------
    x : 2D-array 
        A panel of series (rows: time, columns: series).
    wdw : int
        Length of the MA window length, expressed in number of obs. 
    center : str, optional
        Flag to indicate that the moving windows should be centered with respect 
        to the data. Otherwise, the windows are left-shifted to include only 
        the current and past observations. The default is True.
        
    Returns
    -------
   x_wht_levels : 2D-array
        The data without intrinsic levels.
        
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (n_obs, n_stations) = x.shape
    x_wht_levels = np.copy(x)
    
    assert wdw > 0, "Window length must be a postive integer"
    wdw = np.round(wdw)
        
    if center:
        if wdw % 2 == 1:
            halfwdw = int((wdw - 1)/2)
        else:
            wdw += 1
            halfwdw = int((wdw - 1)/2)
        for i in range(n_stations):
            m = np.nanmean(x[:,i])
            ma = np.ones((n_obs))*m
            for j in range(n_obs):
                if j < halfwdw: #beginning
                    ma[j] = np.nanmean(x[0:wdw,i])
                elif j >= halfwdw and j < n_obs - halfwdw: #middle
                    ma[j] = np.nanmean(x[j-halfwdw :j+halfwdw+1,i])
                else: #end
                    ma[j] = np.nanmean(x[n_obs-wdw :n_obs,i])
            ma[np.isnan(ma)] = m
            x_wht_levels[:,i] = x_wht_levels[:,i] - ma
            
    if not center:
        for i in range(n_stations):
            m = np.nanmean(x[:,i])
            ma = np.ones((n_obs))*m
            for j in range(n_obs):
                if j < wdw: #beginning
                    ma[j] = np.nanmean(x[0:wdw,i]) 
                else: #remaining part
                    ma[j] = np.nanmean(x[j - wdw+1:j+1,i]) 
            ma[np.isnan(ma)] = m
            x_wht_levels[:,i] = x_wht_levels[:,i] - ma
            
        
    return x_wht_levels


def pool_clustering(x, method='kmeans', ref=None, nIC_inf=0.25, nIC=None):
    """
    Selects a subset (also called 'pool') of in-control (IC) processes 
    from a panel.
    
    This function automatically selects the IC processes of a panel of series
    using different clustering methods.    
    A robust version of the mean-squarred error (mse) is calculated for each series
    of the panel. Then, a clustering algorithm groups the mse into two groups: 
    the IC or stable group and out-of-control (OC) or unstable group.
    The function returns the IC group. 
    

    Parameters
    ----------
    x : 2D-array
       A matrix representing a panel of time-series to be clustered.
       (rows: time, columns: series) 
    method : str, optional
         String that designates the clustering method that will be used.
         Values for 'method' should be selected from:
        'leftmed': all series whose mse is inferior to the median of the mse 
        are selected ;
        'kmeans': K-means clustering ;
        'agg': agglomerative hierarchical clustering ;
        'ms': mean-shift clustering ;
        'dbscan': DBscan clustering ;
        'gmm': gaussian mixture models used for clustering ;
        'fix': selects the first 'nIC' most stable processes
        The default is 'kmeans'.
    ref : float, optional
       The 'true' reference of the panel. If None, the median of the network is 
       used as a reference. The default is None.
    nIC_inf : float in [0,1], optional
        Lower bound, expressed in percentage, for the number of IC processes.
        Typical value are 0.25 or 0.3. 
        The number of IC series are computed to be in range [nIC_inf*n_series ; 
        (1-nIC_inf)*n_series], where n_series designates the number of 
        diffrent series. 
        The default is 0.25.
    nIC : int, optional
      Number of IC stations selected (if method='fix'). 
      The default is None.
      
    Returns
    -------
    pool : list
        The indexes of the IC series.     
      
    """
    
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    assert method in ['leftmed','kmeans','agg','ms', 'dbscan','gmm','fix'], "Undefined clustering method"
    assert nIC_inf >= 0 and nIC_inf <= 1, "nIC_inf must be included in [0,1]"
    
    (n_obs, n_series) = x.shape
    if ref is None: 
        ref = np.nanmedian(x)
    if nIC is None:
        nIC = n_series/2
        
    nIC = int(nIC)
    assert nIC > 0, "nIC must be strictly positive"
    

    #compute the robust mse
    mse = np.zeros((n_series))
    for i in range(n_series):
        mse[i] = (np.nanmedian(x[:,i] - ref))**2 + iqr(x[:,i], nan_policy='omit') 
    ordered_stations = np.argsort(mse)
        
    ### Different methods:
    ### All series that are more stable than the median 
    if method == 'leftmed':
        pool = list(np.where(mse < np.median(mse))[0])
        
    ###K-means
    if method == 'kmeans':
        mse_init = np.copy(mse)
        kmeans = KMeans(n_clusters=2, init='k-means++', random_state=100).fit(mse.reshape(-1,1))
        labels_km = kmeans.labels_
        best = labels_km[ordered_stations[0]] #label of the most stable series
        n_best = len(labels_km[labels_km == best])
        worst = labels_km[ordered_stations[-1]] #label of the most stable series
        n_worst = len(labels_km[labels_km == worst])
        pool = [i for i in range(n_series) if labels_km[i] == best]
        
        #too much series into OC group
        while n_worst > (1-nIC_inf)*n_series:
            ind_bad = np.where(labels_km==worst)[0]
            mse = mse[ind_bad]
            ordered_array = np.argsort(mse)
            kmeans = KMeans(n_clusters=2, init='k-means++', random_state=100).fit(mse.reshape(-1,1))
            labels_km = kmeans.labels_
            worst = labels_km[ordered_array[-1]] #label of the worst series
            n_worst = len(labels_km[labels_km == worst])
            pool = [i for i in range(n_series) if mse_init[i] not in mse[labels_km == worst]]
            
        #too much series into IC group
        while n_best > (1-nIC_inf)*n_series: 
            ind_good = np.where(labels_km==best)[0]
            mse = mse[ind_good]
            ordered_array = np.argsort(mse)
            kmeans = KMeans(n_clusters=2, init='k-means++', random_state=100).fit(mse.reshape(-1,1))
            labels_km = kmeans.labels_
            best = labels_km[ordered_array[0]] #label of the most stable series
            n_best = len(labels_km[labels_km == best])
            pool = [i for i in range(n_series) if mse_init[i] in mse[labels_km == best]]
        
    ####################################
    ### Agglomerative clustering
    if method == 'agg':
        mse_init = np.copy(mse)
        agg = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(mse.reshape(-1, 1))
        labels_agg = agg.labels_
        best = labels_agg[ordered_stations[0]]
        n_best = len(labels_agg[labels_agg == best])
        worst = labels_agg[ordered_stations[-1]] 
        n_worst = len(labels_agg[labels_agg == worst])
        pool = [i for i in range(n_series) if labels_agg[i] == best]
        
        #too much series into OC group
        while n_worst > (1-nIC_inf)*n_series:
            ind_bad = np.where(labels_agg==worst)[0]
            mse = mse[ind_bad]
            ordered_array = np.argsort(mse)
            agg = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(mse.reshape(-1, 1))
            labels_agg = agg.labels_
            worst = labels_agg[ordered_array[-1]] #label of the worst series
            n_worst = len(labels_agg[labels_agg == worst])
            pool = [i for i in range(n_series) if mse_init[i] not in mse[labels_agg == worst]]
            
        #too much series into IC group
        while n_best > (1-nIC_inf)*n_series: 
            ind_good = np.where(labels_agg==best)[0]
            mse = mse[ind_good]
            ordered_array = np.argsort(mse)
            agg = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(mse.reshape(-1, 1))
            labels_agg = agg.labels_
            best = labels_agg[ordered_array[0]] #label of the most stable series
            n_best = len(labels_agg[labels_agg == best])
            pool = [i for i in range(n_series) if mse_init[i] in mse[labels_agg == best]]
            
        
    ###########################################"
    ### Gaussian mixture models clustering 
    if method == 'gmm': 
        mse_init = np.copy(mse)
        gmm = GaussianMixture(n_components=2).fit(mse.reshape(-1, 1))
        labels_gmm = gmm.predict(mse.reshape(-1, 1))
        best = labels_gmm[ordered_stations[0]]
        n_best = len(labels_gmm[labels_gmm == best])
        worst = labels_gmm[ordered_stations[-1]] 
        n_worst = len(labels_gmm[labels_gmm == worst])
        pool = [i for i in range(n_series) if labels_gmm[i] == best]
        
        #too much series into OC group
        while n_worst > (1-nIC_inf)*n_series:
            ind_bad = np.where(labels_gmm==worst)[0]
            mse = mse[ind_bad]
            ordered_array = np.argsort(mse)
            gmm = GaussianMixture(n_components=2).fit(mse.reshape(-1, 1))
            labels_gmm = gmm.predict(mse.reshape(-1, 1))
            worst = labels_gmm[ordered_array[-1]] #label of the worst series
            n_worst = len(labels_gmm[labels_gmm == worst])
            pool = [i for i in range(n_series) if mse_init[i] not in mse[labels_gmm == worst]]
            
        #too much series into IC group
        while n_best > (1-nIC_inf)*n_series: 
            ind_good = np.where(labels_gmm==best)[0]
            mse = mse[ind_good]
            ordered_array = np.argsort(mse)
            gmm = GaussianMixture(n_components=2).fit(mse.reshape(-1, 1))
            labels_gmm = gmm.predict(mse.reshape(-1, 1))
            best = labels_gmm[ordered_array[0]] #label of the most stable series
            n_best = len(labels_gmm[labels_gmm == best])
            pool = [i for i in range(n_series) if mse_init[i] in mse[labels_gmm == best]]
            
    ###################################
    ### Mean-shifts
    if method == 'ms':
        bandwidth = estimate_bandwidth(mse.reshape(-1, 1), quantile=0.2, n_samples=len(mse))
        ms = MeanShift(bandwidth).fit(mse.reshape(-1, 1))
        labels_ms = ms.labels_
        best = [] 
        best.append(labels_ms[ordered_stations[0]]) #find label of most stable station
        for i in range(1, n_series):
            n = len(labels_ms[np.isin(labels_ms, best)]) #number of stations with label included into 'best'
            new_label = labels_ms[ordered_stations[i]] #label of next stable station
            if n < nIC_inf*n_series and new_label not in best:
                best.append(new_label) 
        pool = [i for i in range(n_series) if labels_ms[i] in best]

    ####################################
    ### Density-based clustering
    if method == 'dbscan':
        dbscan = DBSCAN(eps=0.01, min_samples=2).fit(mse.reshape(-1, 1))
        labels_db = dbscan.labels_
        best = [] 
        best.append(labels_db[ordered_stations[0]]) 
        for i in range(1, n_series):
            n = len(labels_db[np.isin(labels_db, best)])
            new_label = labels_db[ordered_stations[i]] 
            if n < nIC_inf*n_series and new_label not in best:
                best.append(new_label)
        pool = [i for i in range(n_series) if labels_db[i] in best]
 
    if method == 'fix': 
        pool = list(ordered_stations[:nIC])
            
    return pool



def outliers_removal(data, pool, k=2):
    """
    Removes the outliers from a panel of IC series. 
    
    This function removes at each time the observations of the IC series
    that do not fall into a multiple 'k' of the standard deviation 
    around the mean.
    
    Parameters
    ----------
    data : 2D-array
       A matrix representing a panel of time-series (rows: time, columns: series).
    pool : 1D-array (int)
        An array with the indexes of the in-control (IC) series of a panel.
    k : float, optional
        Multiple of the standard deviation that defines an outlier. 
        Typical values are 1.5, 2 or 3 (depending of the noise of the data) but should 
        be larger than or equal to 1 (otherwise too many IC data are suppressed).
        The default is 2. 
        
    Returns
    -------
    dataIC : 2D-array
        The IC series without outliers.
        
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape
    dataIC = data[:,pool]
    
    for i in range(n_obs):
        avail = np.where(~np.isnan(data[i,:]))
        avail_IC = np.where(~np.isnan(dataIC[i,:]))
        #index = np.where((dataIC[i,:] > np.nanmedian(data[i,:]) + k*iqr(data[i,:], nan_policy='omit')) | (dataIC[i,:]<np.nanmedian(data[i,:]) - k*iqr(data[i,:], nan_policy='omit')))
        index = np.where((dataIC[i,:] > np.nanmean(data[i,:]) + k*np.nanstd(data[i,:])) | (dataIC[i,:] < np.nanmean(data[i,:]) - k*np.std(data[i,:])))
        if ((index is not None) and len(avail[0]) > len(avail_IC[0])*1.1):
            dataIC[i,index] = np.nan


def standardisation(data, dataIC, K):
    """ 
    Standardises the data with time-dependent parameters. 
    
    The time-varying mean and the variance are computed across the in-control
    (IC) series and along a small window of time using K nearest neighbors (KNN) 
    regression estimators.
    
    Parameters
    ----------
    data : 2D-array
        A matrix representing a panel of time-series to be standardised.
        (rows: time, columns: series)
    dataIC : 2D-array
        A subset of the panel containing IC series to be standardised.
        The time-varying mean and variance are estimated on this stable subset.
    K :  int > 0
        Number of nearest neighbors. 
        
    Returns
    -------
    data_stn : 2D-array
        The standardised data.
    dataIC_stn : 2D-array
        The standardised IC data. 
    
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    assert np.ndim(dataIC) == 2, "Input data must be a 2D array"
    assert K > 0, "K must be a postive integer"
    K = np.round(K)
    
    (n_obs, n_stations) = data.shape
    mu_0 = np.nanmean(dataIC); sigma_0 = np.nanstd(dataIC)
    mu_t = np.ones((n_obs))*mu_0; sigma_t = np.ones((n_obs))*sigma_0
    for t in range(n_obs):
        c = 0
        mylist = [] 
        for i in range(n_obs):
            if c < K:
                if t+i < n_obs:
                    data_tb = dataIC[t+i,:]
                    nb = data_tb[~np.isnan(data_tb)]
                    mylist.extend(nb)
                    c += len(nb) 
                if t-i >= 0 and i > 0:
                    data_ta = dataIC[t-i,:]
                    na = data_ta[~np.isnan(data_ta)]
                    mylist.extend(na)
                    c += len(na) 
            if ((c >= K and len(mylist) > 0) or (i == n_obs - 1 and len(mylist) > 0)):  
                mu_t[t] = np.mean(mylist)
                sigma_t[t] = np.std(mylist)
                break
        
    sigma_t[sigma_t == 0] = sigma_0 #don't divide by zero 
        
    data_stn = np.copy(data)
    for i in range(n_stations):
        data_stn[:,i] = (data_stn[:,i] - mu_t)/sigma_t
        
    dataIC_stn = np.copy(dataIC)
    for i in range(dataIC.shape[1]):
        dataIC_stn[:,i] = (dataIC_stn[:,i] - mu_t)/sigma_t  
    
    return data_stn, dataIC_stn


def choice_K(data, dataIC, start=200, stop=10000, step=200, plot=True):
    """ 
    Computes an appropriate value for the number of nearest neighbors (K).
    
    This function evaluates the standard deviation of the standardised data 
    for different values of K. Then, the value where the standard deviation 
    starts to stabilizes ('knee' of the curve) is selected using a 'knee' 
    locator. 
    
    Parameters
    ----------
    data : 2D-array
        A matrix representing a panel of time-series to be standardised
        by K-NN regression estimators. (rows: time, columns: series)
    dataIC : 2D-array
        A subset of the panel containing stable (in-control) series to be 
        standardised.
        The time-varying mean and variance are estimated on this stable subset. 
    start : int > 0, optional
        Lower value for K. The number of nearest neighbours are tested in
        the range [start, stop]. Default is 200.
    stop : int > 0, optional
        Upper value for K. The number of nearest neighbours are tested in
        the range [start, stop]. Default is 10000.
    step : int > 0, optional
        Step value for K. The number of nearest neighbours are tested in 
        the range [start, stop], with step equal to 'step'.
        Default is 200.
    plot : bool, optional 
        Flag to plot the mean and the standard deviation of the standardized 
        data as a function of K. Default is True.
        
    Returns
    ------
    K : int > 0
       The selected number of nearest neighbors.
       
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    assert np.ndim(dataIC) == 2, "Input data must be a 2D array"
    start = int(start); stop = int(stop); step = int(step)
    assert start > 0, "start must be strickly postive"
    assert stop > 0, "stop must be strickly postive"
    assert step > 0, "step must be strickly postive"
    
    n = int(np.ceil((stop - start)/step))
    mean_data = np.zeros((n)); std_data = np.zeros((n))
    c=0 ; x = np.arange(start,stop,step)
    for K in range(start, stop, step):
        #standardise the data with K-NN estimators
        data_stn = standardisation(data, dataIC, K)[0]
        mean_data[c] = np.nanmean(data_stn)
        std_data[c] = np.nanstd(data_stn)
        c += 1
        
    if plot: 
        plt.plot(x, mean_data[:len(x)]);  plt.xlabel('K'); plt.ylabel('mean')
        plt.title('Mean of the data as a function of K'); plt.show()
        plt.plot(x, std_data[:len(x)]); plt.xlabel('K'); plt.ylabel('std')
        plt.title('Std of the data as a function of K'); plt.show()
    
    y = std_data[:len(x)]
    coef = np.polyfit(x, y, deg=1)
    coef_curve = np.polyfit(x, y, deg=2)
    if coef_curve[0] < 0: 
        curve = 'concave'        
    else: 
        curve = 'convex'
    if coef[0] < 0: #slope is positive
        direction = 'decreasing'
    else: #slope is negative
        direction = 'increasing'
    kn = KneeLocator(x, y, curve=curve, direction=direction)
    K = kn.knee 
    
    return K


def smoothing(data, wdw=30, min_perc_per_wdw=10, center=True):
    """ 
    Smooths a panel of data, series by series, using a moving average (MA) filter
    with window length equal to 'wdw'.

    
    Parameters
    ---------
    data : 2D-array 
        A matrix of observations to be smoothed
        (rows: time, columns: stations).
    wdw : int>0, optional
        Length of the MA window, expressed in number of observations. 
        This parameter should be superior than 0. 
        The default is 30.
    min_perc_per_wdw : int, optional
        Minimum percentage of obs required per window to compute a value for
        that day (otherwise NaN). Default is ten percents. 
    center : str, optional
        Flag to indicate that the moving windows should be centered with respect 
        to the data. Otherwise, the windows are left-shifted to include only 
        the current and past observations. The default is True. 
        
    Returns
    ------
    data_sm : 2D-array
        The smoothed data.
        
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_stations) = data.shape
    assert wdw > 0, "Window length must be strictly positive"
        
    data_sm = np.ones((n_obs, n_stations)); data_sm[:] = np.nan
    
    ### smoothing procedure
    if not center:
        for i in range(n_stations):
            for j in range(n_obs):
                if j < wdw: #beginning
                    m = data[0:wdw,i]
                else: #remaining part
                    m = data[j - wdw+1:j+1,i]
                if len(m[~np.isnan(m)]) > np.round(wdw/min_perc_per_wdw):
                    data_sm[j,i] = np.nanmean(m) 
                    
                
    elif center:           
        if wdw % 2 == 1:
            halfwdw = int((wdw - 1)/2)
        else:
            wdw += 1
            halfwdw = int((wdw - 1)/2)
        
        for i in range(n_stations):
            for j in range(n_obs):
                if j < halfwdw: #beginning
                    m = data[0:wdw,i]
                elif j >= halfwdw and j < n_obs - halfwdw: #middle
                    m = data[j - halfwdw:j + halfwdw + 1,i]
                else: #end
                    m = data[n_obs - wdw:n_obs,i]
                if len(m[~np.isnan(m)]) > np.round(wdw/min_perc_per_wdw):
                    data_sm[j,i] = np.nanmean(m) 
                
        
    return data_sm



