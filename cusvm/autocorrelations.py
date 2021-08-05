# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:02:22 2020

@author: sopmathieu

This file contains different functions to compute and analyze the autocorrelation
of time-series that may contain missing values.

These functions compute the autocorrelation and the autocovariance of a series 
at a desired lag. They plot the autocorrelation and partial autocorrelation
functions of a series until a maximum lag. They also calculate the p-values 
associated to the porte-manteau test at each lag.  
Finally, a procedure to automatically select the block length
(of a BB method) is included in this file. 

"""

import numpy as np 
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.utils import resample
from kneed import KneeLocator

def autocorrelation(Xi, k=1):
    """ 
    Computes the autocorrelation at lag k, valid for missing values.
    
    Parameters
    ----------
    Xi : 1D array
        A single time-series.
    k :  int, optional
        The lag at which the autocorrelation should be computed. 
        The default is one. 
        
    Returns
    -------
    autoCorr : float
        The autocorrelation at lag k.
        
    """
    if k >= 1:
        ts1 = Xi[:-k]; ts2 = Xi[k:]
    else: 
        ts1 = Xi; ts2 = Xi
    N = len(ts1)
    Xs1 = np.nanmean(ts1); Xs2 = np.nanmean(ts2)
    autoCov = 0; c=0
    for i in np.arange(0, N):
        if (not np.isnan(ts1[i]) and not np.isnan(ts2[i])):
            autoCov += (ts1[i]-Xs1) * (ts2[i]-Xs2)
            c += 1
    
    autoCorr = ((1/c)*autoCov*(1/(np.nanstd(Xi[:-k])*np.nanstd(Xi[k:]))))
    
    return autoCorr

def autocovariance(Xi, k=1):
    """ 
    Computes the autocovariance at lag k, valid for missing values.
    
    Parameters
    ----------
    Xi : 1D array
        A single time-series.
    k :  int, optional
        The lag at which the autocovariance should be computed. 
        The default is one. 
        
    Returns
    -------
    autoCov : float
        The autocovariance at lag k.
        
    """
    if k >= 1:
        ts1 = Xi[:-k]; ts2 = Xi[k:]
    else: 
        ts1 = Xi; ts2 = Xi
    N = len(ts1)
    Xs1 = np.nanmean(ts1); Xs2 = np.nanmean(ts2)
    autoCov = 0; c=0
    for i in np.arange(0, N):
        if (not np.isnan(ts1[i]) and not np.isnan(ts2[i])):
            autoCov += (ts1[i]-Xs1) * (ts2[i]-Xs2)
            c += 1
    return (1/c)*autoCov


def autocorr(x, k=1):
    """ 
    Computes the autocorrelation at lag k, valid for missing values
    and faster than previous function.
    
    Parameters
    ----------
    x : 1D array
        A single time-series.
    k :  int, optional
        The lag at which the autocorrelation should be computed. 
        The default is one. 
        
    Returns
    -------
    autoCorr : float
        The autocorrelation at lag k.
        
    """
    if k >= 1:
        ts1 = x[:-k]; ts2 = x[k:]
    else: 
        ts1 = x; ts2 = x
    a = np.ma.masked_invalid(ts1)
    b = np.ma.masked_invalid(ts2)
    msk = (~a.mask & ~b.mask)
    
    autoCorr = (np.corrcoef([a[msk], b[msk]]))[0,1]
    return autoCorr

    
def autocov(x, k=1):
    """ 
    Computes the autocovariance at lag k, valid for missing values.
    
    Parameters
    ----------
    x : 1D array
        A single time-series.
    k :  int, optional
        The lag at which the autocovariance should be computed. 
        The default is one. 
        
    Returns
    -------
    autoCov : float
        The autocovariance at lag k.
        
    """
    if k >= 1:
        ts1 = x[:-k]; ts2 = x[k:]
    else: 
        ts1 = x; ts2 = x
    a = np.ma.masked_invalid(ts1)
    b = np.ma.masked_invalid(ts2)
    msk = (~a.mask & ~b.mask)
    
    autoCov =  (np.cov([a[msk], b[msk]], ddof=0))[0][1] #otherwise different normalization
    
    return autoCov
 
#================================================================
### plots 
#===============================================================
    
def acf_pacf_plot(x, which_display=0, max_cov=50):
    """ 
    Plots the autocorrelation function (acf) and the partial 
    autocorrelation function (pacf) for a time-series 
    with missing observations (except at lag=0).
    
    Parameters
    ----------
    x : 2D-array
       A panel of time-series. 
    which_display : int, optional
        The index of the series in the panel to be displayed (acf, pacf).
        The default is zero.
    max_cov : int>0, optional
        The maximum lag until the autocovariance should 
        be computed. The defaults is 50.
        
    Returns
    -------
    fig : a matplotlib figure
        The figure with the acf and pacf.
        
    """
    (row_x, column_x) = x.shape
    corr_data = np.zeros((column_x, max_cov + 1)) 
    p_ljung = np.zeros((column_x, max_cov))
    partcorr_data = np.zeros((column_x, max_cov + 1)) 
    for i in range(column_x):
        if np.count_nonzero(~np.isnan(x[:,i])) > max_cov+1:
            intm = acf(x[:,i], nlags=max_cov, missing='drop',alpha=0.05, qstat='True')
            corr_data[i,:] = intm[0]
            p_ljung[i,:] = intm[3]
            x_wht_nan = x[:,i]
            intm = pacf(x_wht_nan[~np.isnan(x_wht_nan)], max_cov, alpha=0.05)
            partcorr_data[i,:] = intm[0] 

    display = x[:,which_display] #which series to display
    display = display[~np.isnan(display)] #remove nans
    ci = np.ones(max_cov) *stats.norm.ppf((1 + 0.95)/2)/np.sqrt(len(display))
    
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    fig = plt.figure()
    f1 = plt.subplot(2,1,1)
    plt.stem(np.arange(max_cov)+1, corr_data[which_display,1:], basefmt='k', use_line_collection=True)   
    plt.plot(np.arange(max_cov)+1, ci,'k:')  
    plt.plot(np.arange(max_cov)+1, -ci,'k:') 
    plt.fill_between(np.arange(max_cov)+1, -ci, ci,color='b', alpha=0.2)
    plt.title("Autocorrelation function in series %s" %which_display)
    #plt.axis([-2, max_cov+2, -0.2, 1.05])
    f1.set_xlim([-2, max_cov+2])
    
    f2 = plt.subplot(2,1,2)
    plt.stem(np.arange(max_cov)+1, partcorr_data[which_display,1:], basefmt='k',use_line_collection=True)   
    plt.plot(np.arange(max_cov)+1, ci,'b:')  
    plt.plot(np.arange(max_cov)+1, -ci,'b:') 
    plt.fill_between(np.arange(max_cov)+1, -ci, ci,color='b', alpha=0.2)
    plt.title("Partial-Autocorrelation function in series %s" %which_display)
    #plt.axis([-2, max_cov+2, -0.2, 1.05])
    f2.set_xlim([-2, max_cov+2])
    plt.show()
    
    return fig


def acf_pacf_residuals(res, max_cov=50):
    """ 
    Plots the autocorrelation function (acf), the partial 
    autocorrelation function (pacf) and the p-values of the 
    lung-box test for residuals of ARMA models. 
    
    Parameters
    ----------
    res : 1D-array
       The residuals of an ARMA model.
    max_cov : int>0, optional
        The maximum lag until the autocovariance should 
        be computed. The defaults is 50.
        
    Returns
    -------
    fig : a matplotlib figure
        The figure with the acf, pacf and the p-values.
    """
    
    n = len(res)
    acf_res = acf(res, nlags=max_cov, qstat='True')[0]
    p_value_res = acf(res, nlags=max_cov, qstat='True')[2]
    pacf_res = pacf(res, nlags=max_cov)

    fig = plt.figure()
    ci = np.ones(max_cov) * stats.norm.ppf((1 + 0.95)/2)/np.sqrt(n)
    plt.subplot(3,1,1)
    plt.stem(np.arange(max_cov)+1, acf_res[1:], basefmt='k')   
    plt.plot(np.arange(max_cov)+1, ci,'k:')  
    plt.plot(np.arange(max_cov)+1, -ci,'k:') 
    plt.fill_between(np.arange(max_cov), -ci, ci,color='b', alpha=0.2)
    plt.title("Acf of residuals")
    plt.axis([-2, max_cov+2, -0.1, 1.2])
    plt.subplot(3,1,2)
    plt.stem(np.arange(max_cov)+1, pacf_res[1:], basefmt='k')   
    plt.plot(np.arange(max_cov)+1, ci,'b:')  
    plt.plot(np.arange(max_cov)+1, -ci,'b:') 
    plt.fill_between(np.arange(max_cov), -ci, ci,color='b', alpha=0.2)
    plt.title("Pacf of residuals")
    plt.axis([-2, max_cov+2, -0.1, 1.2])
    plt.subplot(3,1,3)
    plt.plot(np.arange(max_cov)+1, p_value_res,'o')  
    plt.plot(np.arange(max_cov)+1, np.ones(max_cov)*0.05,'b:')  
    plt.title("p-values of the Ljung-box chi squared stats")
    plt.axis([-2, max_cov+2, -0.1, 1.2])
    plt.show()
    
    return fig

#================================================================
### choice of the block length
#===============================================================


def block_length_choice(data, bbl_min=10, bbl_max=110, bbl_step=10, 
                       n_corr=50, nmc=200, BB_method='MBB', plot=True):
    """
    Computes an appropriate value of the block length for a panel of 
    time-series with missing values.
    
    The algorithm works as follows.
    For each block length tested over the specified range, this function resamples
    several series of observations using a block bootstrap procedure.
    Then, it computes the mean squared error (MSE) of the mean,
    standard deviation and autocorrelation at different lags 
    of the resampled series (with respect to the original data). 
    Small block lengths represent the variance and the mean of the data properly 
    (mse of the mean and the variance increases when block length augments).
    Whereas large block lengths better account for the autocorrelation of the data
    (mse of the autocorrelation diminishes when block length increases). 
    The appropriate value for the block length is finally selected as the first
    value such that the mse of the autocorrelation stabilizes
    ("knee" of the curve).
    This value intuitively corresponds to the smallest length which is 
    able to represent the main part of the autocorrelation of the series.

    Parameters
    ----------
    data : 2D-array
        A matrix representing a panel of time-series to be resampled by
        a block boostrap procedure. (rows: time, columns: series)
        To save computational time, only the IC series may be used. 
    bbl_min : int, optional
        Lower value for the block length. The block lengths are tested in
        the range [bbl_min, bbl_max]. Default is 10.
    bbl_max : int, optional
        Upper value for the block length. The block lengths are tested in
        the range [bbl_min, bbl_max]. Default is 110.
    bbl_step : int, optional
        Step value for the block length. The block lengths are tested in 
        the range [bbl_min, bbl_max], with step equal to bbl_step.
        Default is 10.
    n_corr : int, optional
        Maximal lag up to which the autocorrelation is evaluated.
        The default is 50.
    nmc  : int > 0, optional
        Number of resampled series used to compute the MSEs.
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       If the matched block bootstrap is intended to be use, 'NBB' may be 
       selected (the matched block bootstrap is based on the 'NBB'). 
       'CBB': circular block bootstrap
       Default is 'MBB'.
    plot : bool, optional
       Flag to show the figures (and print some results). The default is True. 
      
    Returns
    -------
    block-length : int
        The selected block length. 

    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape
    assert BB_method in ['MBB', 'NBB', 'CBB'], "Undefined block bootstrap procedure"
    
    #compute the autocorrelation of the initial data
    corr_data = np.zeros((n_series, n_corr))  
    for j in range(n_series):
        for i in range(n_corr):
            corr_data[j,i] = autocorr(data[:,j],i+1)
      
    ### parameters 
    n_bbl = int(np.ceil((bbl_max - bbl_min)/bbl_step)) #number of block length sizes tested
    mse_mean_series = np.zeros((n_bbl, n_series)); mse_std_series = np.zeros((n_bbl, n_series))
    mse_corr_lag = np.zeros((n_corr, n_series)) ; mse_corr_series = np.zeros((n_bbl, n_series))
    bbl = np.zeros(n_bbl)
    c = 0
    for block_length in range(bbl_min, bbl_max, bbl_step): 
        
        ### Create blocks (moving block bootstrap)
        bbl[c] = block_length
        n_blocks = int(np.ceil(n_obs/block_length))
        
        if BB_method == 'MBB':
            N = n_obs - block_length + 1
            blocks = np.zeros((N, block_length, n_series)) 
            for j in range(n_series):
                for i in range(N):
                    blocks[i,:,j] = data[i:i+block_length, j] #series by series
        elif BB_method == 'NBB':
            N = int(np.floor(n_obs / block_length))
            blocks = np.zeros((N, block_length, n_series))
            for j in range(n_series):
                cc = 0
                it = 0
                for i in range(0, N):
                    blocks[cc,:,j] = data[it:it+block_length,j] #non-overlapping
                    it += block_length
                    cc += 1
        elif BB_method == 'CBB':
                N = n_obs
                blocks = np.zeros((N, block_length, n_series))
                for j in range(n_series): 
                    cc = 0
                    data_dup = np.concatenate((data[:,j], data[:,j]))
                    for i in range(0, N):
                        blocks[cc,:,j] = data_dup[i:i+block_length] #overlapping
                        cc += 1
       
        for j in range(n_series):
            corr_boot = np.zeros((n_corr, nmc)); mean_boot = np.zeros((nmc)); std_boot = np.zeros((nmc))
            corr_boot[:] = np.nan; mean_boot[:] = np.nan; std_boot[:] = np.nan
            
            for b in range(nmc):   
                boot = resample(blocks[:,:,j], replace=True, n_samples=n_blocks).flatten()
                #boot = boot[~np.isnan(boot)]
                for i in range(n_corr):
                    corr_boot[i,b] = autocorr(boot, i+1)  
                mean_boot[b] = np.nanmean(boot)
                std_boot[b] = np.nanstd(boot)
        
            ### results per station
            mse_mean_series[c,j] = (np.nanmean(mean_boot) - np.nanmean(data[:,j]))**2 + np.nanvar(mean_boot)
            mse_std_series[c,j] = (np.nanmean(std_boot) - np.nanstd(data[:,j]))**2 + np.nanvar(std_boot)
            for i in range(n_corr):
                mse_corr_lag[i,j] = (np.nanmean(corr_boot[i,:]) - corr_data[j,i])**2 + np.nanvar(corr_boot[i,:])
            mse_corr_series[c,j] = np.nanmean(mse_corr_lag, axis=0)[j]
        c += 1
    
    #for all stations
    mse_mean = np.nanmean(mse_mean_series, axis=1)
    mse_std = np.nanmean(mse_std_series, axis=1)  
    mse_corr = np.nanmean(mse_corr_series, axis=1) 
        
    x = bbl
    y = mse_corr
    
    
    #select the knee of the curve
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
    block_length = kn.knee 
    
    if plot: 
        plt.rcParams['figure.figsize'] = (10.0, 6.0)
        plt.rcParams['font.size'] = 14
        plt.plot(x, y, marker='o'); plt.xlabel('block length')
        plt.ylabel('mse of autocorrelation')
        plt.title('MSE of the autocorrelation as function of the block length')
        if block_length is not None:
            plt.axvline(x=block_length, color='orange', linestyle='--', label='selected value \n (knee)')
            plt.legend()
        plt.show()
        print('Block length which minimizes the mse of the mean:', x[np.argmin(mse_mean)])#0
        print('Block length which minimizes the mse of the std:', x[np.argmin(mse_std)])#0
        print('Block length which minimizes the mse of the autocorrelation:', x[np.argmin(mse_corr)]) #100
    
    return block_length


#############################################################################"    
""" Tests """

if __name__ == "__main__":
    
    x = [-2.1, -1,  4.3] 
    k = 1
    inte = (x[k:]-np.nanmean(x[k:])) * (x[:-k]-np.nanmean(x[:-k]))
    cov_x = np.nanmean(inte)
    corr_x = np.nanmean(inte) * (1/(np.nanstd(x[k:]) * np.nanstd(x[:-k])))
    corr_test1_x = autocorrelation(x)
    corr_test2_x = autocorr(x)
    cov_test1_x = autocovariance(x)
    cov_test2_x = autocov(x)
    
    y = 20 * np.random.randn(1000) + 100
    k = 1
    inte = (y[k:]-np.nanmean(y[k:])) * (y[:-k]-np.nanmean(y[:-k]))
    cov_y = np.nanmean(inte)
    corr_y = np.nanmean(inte) * (1/(np.nanstd(y[k:]) * np.nanstd(y[:-k])))
    corr_test1_y = autocorrelation(y)
    corr_test2_y = autocorr(y)
    cov_test1_y = autocovariance(y)
    cov_test2_y = autocov(y)
    
    
    ##### autoregressive models 
    phi1 = 0.9
    ar1 = np.array([1, phi1])
    ma1 = np.array([1])
    AR_object1 = ArmaProcess(ar1, ma1)
    simulated_data_1 = AR_object1.generate_sample(nsample=1000)
    
    ar2 = np.array([1, -0.9])#inverse sign 
    ma2 = np.array([1])
    AR_object2 = ArmaProcess(ar2, ma2)
    simulated_data_2 = AR_object2.generate_sample(nsample=1000) 
    
    
    n_cov = 40
    corr_ar1=np.zeros((n_cov)) ; corr_ar2=np.zeros((n_cov)) 
    cov_ar1=np.zeros((n_cov)) ; cov_ar2=np.zeros((n_cov)) 
    cov_test1=np.zeros((n_cov)) ; cov_test2=np.zeros((n_cov))    
    corr_test1=np.zeros((n_cov)) ; corr_test2=np.zeros((n_cov))  
    for i in range(n_cov):
        corr_ar1[i] = autocorr(simulated_data_1,i+1)   
        corr_ar2[i] = autocorr(simulated_data_2,i+1) 
        cov_ar1[i] = autocov(simulated_data_1,i+1)   
        cov_ar2[i] = autocov(simulated_data_2,i+1) 
        
        corr_test1[i] = autocorrelation(simulated_data_1, i+1)   
        corr_test2[i] = autocorrelation(simulated_data_2, i+1) 
        cov_test1[i] = autocovariance(simulated_data_1, i+1)   
        cov_test2[i] = autocovariance(simulated_data_2, i+1) 
        
    simulated_data_1 = simulated_data_1.reshape(-1,1)
    plot_ar1 = acf_pacf_plot(simulated_data_1)
    
    simulated_data_2 = simulated_data_2.reshape(-1,1)
    plot_ar2 = acf_pacf_plot(simulated_data_2)