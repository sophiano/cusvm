# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:30:50 2020

@author: sopmathieu

This file contains several functions to calibrate a CUSUM chart
for a panel of time series subject to missing values. 

It includes functions to compute the control limit, in-control and 
out-of-control average run lengths of the chart. 

"""

import numpy as np 
from sklearn.utils import resample

from cusvm import bb_methods as bb


def ARL0_CUSUM(data, L_plus, L_minus=None, delta=1.5, k=None, nmc=4000,
                  n=4000, two_sided=True, missing_values='omit', 
                  gap=0, block_length=None, BB_method='MBB'):
    """ 
    Computes the in-control (IC) average run length (ARL0) of the CUSUM 
    chart in presence of missing values.
    
    The algorithm works as follows.
    For each monte-carlo run, a new series of observations is sampled from the 
    IC data using a block boostrap procedure. Then, the run length of the chart 
    is evaluated. Finally, the average run length is calculated over the runs.
    
    Parameters
    ---------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    L_plus : float 
        Value for the positive control limit.
    L_minus : float, optional
        Value for the negative control limit. Default is None. 
        When None, L_minus = - L_plus. 
    delta : float, optional
        The target shift size. Default is 1.5. 
    k : float, optional
        The allowance parameter 
        When None, k = delta/2 (optimal formula for iid normal data).
        The default is None.
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    two_sided : bool, optional
        Flag to use two-sided CUSUM chart. Otherwise, the one-sided 
        upper CUSUM chart is used. Default is True.
    missing_values : str, optional
        String that indicates how to deal with the missing values (MV). 
        The string value should be chosen among: 'omit', 'reset' and 'fill':
        'omit' removes the blocks containing MV ;
        'fill' fills-up the MV by the mean of each series ;
        'reset' resets the chart statistics at zero for gaps larger than
        a specified gap length (argument 'gap'). 
        The chart statistics is simply propagated through smaller gaps. 
        Default is 'omit'.
    gap :  int >= 0, optional
        The length of the gaps above which the chart statistics are reset,
        expressed in number of obs. Default is zero. 
    block_length :  int > 0, optional
        The length of the blocks. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    
    Returns
    -------
    ARL :  float          
        The IC average run length (ARL0).    
        
    """  
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape    
    assert missing_values in ['fill', 'reset', 'omit'], "Undefined value for 'missing_values'"
    if missing_values == 'fill':
        for i in range(n_series):
            data[np.isnan(data[:,i]),i] = np.nanmean(data[:,i]) #fill obs by the mean of the series
            
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if missing_values == 'fill' or missing_values == 'omit':
        if BB_method == 'MBB': 
            blocks = bb.MBB(data, block_length)    
        elif BB_method == 'NBB':
            blocks = bb.NBB(data, block_length) 
        elif BB_method == 'CBB':
            blocks = bb.CBB(data, block_length) 
        
    else:
        if BB_method == 'MBB': 
            blocks = bb.MBB(data, block_length, NaN=True)    
        elif BB_method == 'NBB':
            blocks = bb.NBB(data, block_length, NaN=True) 
        elif BB_method == 'CBB':
            blocks = bb.CBB(data, block_length, NaN=True)
   
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))
      
    #chart parameters
    if k is None:
        k = abs(delta)/2
    if L_minus is None:
        L_minus = -L_plus
    n = int(n)
    assert n > 0, "n must be strictly positive"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"

    RL_minus = np.zeros((nmc,1)); RL_plus = np.zeros((nmc,1)) 
    RL_minus[:] = np.nan; RL_plus[:] = np.nan
    for j in range(nmc):
        
        if BB_method == 'MABB': 
            boot = bb.resample_MatchedBB(data, block_length, n=n)
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
    
        ### Monitoring ###
        C_plus = np.zeros((n,1)) 
        cp = 0; nan_p = 0
        for i in range(1, n):
            if not np.isnan(boot[i]):
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                cp = 0
            elif (np.isnan(boot[i]) and cp < gap):
                C_plus[i] = C_plus[i-1]
                cp += 1; nan_p += 1
            else: 
                C_plus[i] = 0 
                nan_p += 1
            if C_plus[i] > L_plus:
                RL_plus[j] = i #-nan_p
                break 
            
        C_minus = np.zeros((n,1))
        cm = 0; nan_m = 0
        for i in range(1, n):
            if not np.isnan(boot[i]):
                 C_minus[i] = min(0, C_minus[i-1] + boot[i] + k)
                 cm = 0
            elif (np.isnan(boot[i]) and cm < gap):
                C_minus[i] = C_minus[i-1]
                cm += 1; nan_m += 1
            else: 
                C_minus[i] = 0 
                nan_m += 1
            if C_minus[i] < L_minus:
                RL_minus[j] = i #-nan_m
                break 
    
        if np.isnan(RL_plus[j]):
            RL_plus[j] = n #-nan_p
        if np.isnan(RL_minus[j]):
            RL_minus[j] = n  #-nan_m
           
    if two_sided:
        ARL = (1/(np.mean(RL_minus)) + 1/(np.mean(RL_plus)))**(-1) 
    else: 
        ARL = np.mean(RL_plus)
    return ARL


def limit_CUSUM(data, delta=1.5, k=None, ARL0_threshold=200, rho=2, L_plus=20, 
                    L_minus=0, nmc=4000, n=4000, two_sided=True, verbose=True,
                    missing_values='omit', gap=0, block_length=None, BB_method='MBB'):
    """ 
   Computes the control limit of the CUSUM chart in presence of missing values.
    
   The control limits of the chart are adjusted by a searching algorithm as follows.
   From initial values of the control limit, the actual IC average run 
   length (ARL0) is computed on 'nmc' processes that are sampled with repetition 
   from the IC data by the block bootstrap procedure.
   If the actual ARL0 is inferior (resp. superior) to the pre-specified ARL0, 
   the control limit of the chart is increased (resp. decreased).
   This algorithm is iterated until the actual ARL0 reaches the pre-specified ARL0
   at the desired accuracy.
    
    Parameters
    ---------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    delta : float, optional
        The target shift size. Default is 1.5.
    k : float, optional
        The allowance parameter.  The default is None.
        When None, k = delta/2 (optimal formula for iid normal data).
    ARL0_threshold : int > 0, optional
        Pre-specified value for the IC average run length (ARL0). 
        This value is inversely proportional to the rate of false positives.
        Typical values are 100, 200 or 500. Default is 200.
    rho : float > 0, optional
        Accuracy to reach the pre-specified value for ARL0: 
        the algorithm stops when |ARL0-ARL0_threshold| < rho.
        The default is 2.
    L_plus : float, optional
        Upper value for the positive control limit. Default is 60.
    L_minus : float, optional
        Lower value for the positive control limit. Default is 0. 
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    two_sided : bool, optional
        Flag to use two-sided CUSUM chart. Otherwise, the one-sided 
        upper CUSUM chart is used. Default is True.
    Verbose : bool, optional
        Flag to print intermediate results. Default is True.
    missing_values : str, optional
        String that indicates how to deal with the missing values (MV). 
        The string value should be chosen among: 'omit', 'reset' and 'fill':
        'omit' removes the blocks containing MV ;
        'fill' fills-up the MV by the mean of each series ;
        'reset' resets the chart statistics at zero for gaps larger than
        a specified gap length (argument 'gap'). 
        The chart statistics is simply propagated through smaller gaps. 
        Default is 'omit'.
    gap :  int >= 0, optional
        The length of the gaps above which the chart statistics are reset,
        expressed in number of obs. Default is zero. 
    block_length :  int > 0, optional
        The length of the blocks. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    
    Returns
    ------
    L : float
       The positive control limit of the chart (with this algorithm,
       it has the same value as the negative control limit, with opposite sign). 
       
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape  
    assert missing_values in ['fill', 'reset', 'omit'], "Undefined value for 'missing_values'"
    if missing_values == 'fill':
        for i in range(n_series):
            data[np.isnan(data[:,i]),i] = np.nanmean(data[:,i]) #fill obs by the mean of the series
           
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if missing_values == 'fill' or missing_values == 'omit':
        if BB_method == 'MBB': 
            blocks = bb.MBB(data, block_length)    
        elif BB_method == 'NBB':
            blocks = bb.NBB(data, block_length) 
        elif BB_method == 'CBB':
            blocks = bb.CBB(data, block_length) 
        
    else:
        if BB_method == 'MBB': 
            blocks = bb.MBB(data, block_length, NaN=True) #all_NaN=False   
        elif BB_method == 'NBB':
            blocks = bb.NBB(data, block_length, NaN=True) 
        elif BB_method == 'CBB':
            blocks = bb.CBB(data, block_length, NaN=True)
   
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))
        
        
    #chart parameters
    if k is None:
        k = abs(delta)/2
    assert L_plus > L_minus, "L_plus should be superior than L_minus"
    L = (L_plus+L_minus)/2
    n = int(n)
    assert n > 0, "n must be strictly positive"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"
    assert rho > 0, "rho must be strictly positive"
    assert ARL0_threshold > 0, "ARL0_threshold must be strictly positive"

    ARL = 0
    while (np.abs(ARL - ARL0_threshold) > rho):
        RL_minus = np.zeros((nmc,1)); RL_plus = np.zeros((nmc,1)) 
        RL_minus[:] = np.nan; RL_plus[:] = np.nan
        for j in range(nmc):
            
            if BB_method == 'MABB': 
                boot = bb.resample_MatchedBB(data, block_length, n=n)
            else:
                boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
    
            
            ### Monitoring ###
            C_plus = np.zeros((n,1)) 
            cp = 0; nan_p = 0
            for i in range(1, n):
                if not np.isnan(boot[i]):
                    C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                    cp = 0
                elif (np.isnan(boot[i]) and cp < gap):
                    C_plus[i] = C_plus[i-1]
                    cp += 1; nan_p += 1
                else: 
                    C_plus[i] = 0 
                    nan_p += 1
                if C_plus[i] > L:
                    RL_plus[j] = i #-nan_p
                    break 
                
            C_minus = np.zeros((n,1))
            cm = 0; nan_m = 0
            for i in range(1, n):
                if not np.isnan(boot[i]):
                     C_minus[i] = min(0, C_minus[i-1] + boot[i] + k)
                     cm = 0
                elif (np.isnan(boot[i]) and cm < gap):
                    C_minus[i] = C_minus[i-1]
                    cm += 1; nan_m += 1
                else: 
                    C_minus[i] = 0 
                    nan_m += 1
                if C_minus[i] < -L:
                    RL_minus[j] = i #- nan_m
                    break 
        
            if np.isnan(RL_plus[j]):
                RL_plus[j] = n #- nan_p
            if np.isnan(RL_minus[j]):
                RL_minus[j] = n #- nan_m
                  
        if two_sided:
            ARL = (1/(np.mean(RL_minus)) + 1/(np.mean(RL_plus)))**(-1) 
        else: 
            ARL = np.mean(RL_plus)
        if ARL < ARL0_threshold:
            L_minus = (L_minus + L_plus)/2
        elif ARL > ARL0_threshold:
            L_plus = (L_minus + L_plus)/2
        L = (L_plus + L_minus)/2
            
        if verbose:
            print(ARL)
            print(L)
            
    return L

def ARL1_CUSUM(data, L_plus, L_minus=None, form='jump', delta=1.5, k=None, 
                  nmc=4000, n=4000, two_sided=True,  missing_values='omit', 
                  gap=0, block_length=None, BB_method='MBB'):
    """ 
    Computes the out-of-control (OC) average run length (ARL1)
    of the CUSUM chart.
    
    The algorithm works as follows.
    For each monte-carlo run, a new series of observations is sampled from the 
    IC data using a block boostrap procedure. Then, a shift of specified form 
    and size is simulated on top of the sample. 
    The run length of the chart is then evaluated. 
    Finally, the average run length is calculated over the runs.
    
    Parameters
    ----------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    L_plus : float 
        Value for the positive control limit.
    L_minus : float, optional
        Value for the negative control limit. Default is None. 
        When None, L_minus = - L_plus. 
    form :  str, optional
         String that represents the form of the shifts that are simulated. 
         The value of the string should be chosen among: 'jump', 'oscillation' 
         or 'drift'.
         Default is 'jump'.
    delta : float, optional
        The target shift size. Default is 1.5.
    k : float, optional
        The allowance parameter (default is None). 
        When None, k = delta/2 (optimal formula for iid normal data).
        The default is None.
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    two_sided : bool, optional
        Flag to use two-sided CUSUM chart. Otherwise, the one-sided 
        upper CUSUM chart is used. Default is True.
    missing_values : str, optional
        String that indicates how to deal with the missing values (MV). 
        The string value should be chosen among: 'omit', 'reset' and 'fill':
        'omit' removes the blocks containing MV ;
        'fill' fills-up the MV by the mean of each series ;
        'reset' resets the chart statistics at zero for gaps larger than
        a specified gap length (argument 'gap'). 
        The chart statistics is simply propagated through smaller gaps. 
        Default is 'omit'.
    gap :  int >= 0, optional
        The length of the gaps above which the chart statistics are reset,
        expressed in number of obs. Default is zero. 
    block_length :  int > 0, optional
        The length of the blocks. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    
    Returns
    ------
    ARL1 : float            
         The OC average run length (ARL1) of the chart.
         
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape 
    assert missing_values in ['fill', 'reset', 'omit'], "Undefined value for 'missing_values'"
    if missing_values == 'fill':
        for i in range(n_series):
            data[np.isnan(data[:,i]),i] = np.nanmean(data[:,i]) #fill obs by the mean of the series
        
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if missing_values == 'fill' or missing_values == 'omit':
        if BB_method == 'MBB': 
            blocks = bb.MBB(data, block_length)    
        elif BB_method == 'NBB':
            blocks = bb.NBB(data, block_length) 
        elif BB_method == 'CBB':
            blocks = bb.CBB(data, block_length) 
        
    else:
        if BB_method == 'MBB': 
            blocks = bb.MBB(data, block_length, NaN=True)    
        elif BB_method == 'NBB':
            blocks = bb.NBB(data, block_length, NaN=True) 
        elif BB_method == 'CBB':
            blocks = bb.CBB(data, block_length, NaN=True)
   
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))
         
    #parameters
    assert form in ['jump','drift','oscillation'], "Undefined shift form"
    shift = delta
    if k is None:
        k = abs(delta)/2
    if L_minus is None:
        L_minus = -L_plus
    n = int(n)
    assert n > 0, "n must be strictly positive"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"

    RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
    RL1_plus[:] = np.nan; RL1_minus[:] = np.nan
    for b in range(nmc):
        
        if BB_method == 'MABB': 
            boot = bb.resample_MatchedBB(data, block_length, n=n)
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
    
        if form == 'oscillation':
            eta = np.random.uniform(0.02, 0.2)
            boot = np.sin(eta*np.pi*np.arange(n))*shift + boot
            pass
        elif form == 'drift':
            power = np.random.uniform(1.5, 2)
            boot = shift/(500)*(np.arange(n)**power) + boot
            pass
        else: 
            boot = boot + shift
            pass

        C_plus = np.zeros((n,1)) 
        cp = 0; nan_p = np.zeros((n,1))
        for i in range(1, n):
            if not np.isnan(boot[i]):
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                cp = 0
            elif (np.isnan(boot[i]) and cp < gap):
                C_plus[i] = C_plus[i-1]
                cp += 1; nan_p[i] = 1
            else: 
                C_plus[i] = 0
                nan_p[i] = 1
            if C_plus[i] > L_plus: 
                ind = nan_p[:i]
                RL1_plus[b] = i #-sum(ind)
                break 
            
        C_minus = np.zeros((n,1)) 
        cm = 0; nan_m = np.zeros((n,1))      
        for j in range(1, n):
            if not np.isnan(boot[j]):
                C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                cm = 0
            elif (np.isnan(boot[j]) and cm < gap):
                C_minus[j] = C_minus[j-1]
                cm += 1; nan_m[j] = 1
            else: 
                C_minus[j] = 0
                nan_m[j] = 1
            if C_minus[j] < L_minus:
                ind = nan_m[:j]
                RL1_minus[b] = j #-sum(ind)
                break
            
        if np.isnan(RL1_plus[b]): 
            RL1_plus[b] = n
        if np.isnan(RL1_minus[b]):
            RL1_minus[b] = n
            
           
    if two_sided:
        ARL1 = (1/(np.nanmean(RL1_minus)) + 1/(np.nanmean(RL1_plus)))**(-1) 
    else: 
        ARL1 = np.mean(RL1_plus)
    return ARL1


def ARL_values(data, L_plus, L_minus=None, form='jump', delta=1.5, k=None,
                  nmc=4000, n=8000, two_sided=True, missing_values='omit',
                  gap=0, block_length=None, BB_method='MBB'):
    """ 
    Computes the in-control (IC) and out-of-control (OC) average run lengths 
    (ARL0 and ARL1) of the CUSUM chart.
    
    The algorithm works as follows.
    For each monte-carlo run, a new series of observations is sampled from the 
    IC data using a block boostrap procedure. The IC run length of the chart 
    is then evaluated.
    A shift of specified form and size is also simulated on top of the sample
    and OC run length of the chart is computed. 
    Finally, the OC and IC average run lengths are calculated over the runs.
    
    Parameters
    ---------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    L_plus : float 
        Value for the positive control limit.
    L_minus : float, optional
        Value for the negative control limit. Default is None. 
        When None, L_minus = - L_plus. 
    form :  str, optional
         String that represents the form of the shift that are simulated. 
         The value of the string should be chosen among: 'jump', 'oscillation'
         or 'drift'.
         Default is 'jump'.
    delta : float, optional
        The target shift size. Default is 1.5.
    k : float, optional
        The allowance parameter (default is None). 
        When None, k = delta/2 (optimal formula for normal data).
        The default is None.
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    two_sided : bool, optional
        Flag to use two-sided CUSUM chart. Otherwise, the one-sided 
        upper CUSUM chart is used. Default is True.
    missing_values : str, optional
        String that indicates how to deal with the missing values (MV). 
        The string value should be chosen among: 'omit', 'reset' and 'fill':
        'omit' removes the blocks containing MV ;
        'fill' fills-up the MV by the mean of each series ;
        'reset' resets the chart statistics at zero for gaps larger than
        a specified gap length (argument 'gap'). 
        The chart statistics is simply propagated through smaller gaps. 
        Default is 'omit'.
    gap :  int >= 0, optional
        The length of the gaps above which the chart statistics are reset,
        expressed in number of obs. Default is zero. 
    block_length :  int > 0, optional
        The length of the blocks. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
  
    Returns
    --------
    ARL1, ARL0: float
       The OC and IC average run lengths (ARL1 and ARL0) of the chart.
       
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape  
    assert missing_values in ['fill', 'reset', 'omit'], "Undefined value for 'missing_values'"
    if missing_values == 'fill':
        for i in range(n_series):
            data[np.isnan(data[:,i]),i] = np.nanmean(data[:,i]) #fill obs by the mean of the series
        
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if missing_values == 'fill' or missing_values == 'omit':
        if BB_method == 'MBB': 
            blocks = bb.MBB(data, block_length)    
        elif BB_method == 'NBB':
            blocks = bb.NBB(data, block_length) 
        elif BB_method == 'CBB':
            blocks = bb.CBB(data, block_length) 
        
    else:
        if BB_method == 'MBB': 
            blocks = bb.MBB(data, block_length, NaN=True)    
        elif BB_method == 'NBB':
            blocks = bb.NBB(data, block_length, NaN=True) 
        elif BB_method == 'CBB':
            blocks = bb.CBB(data, block_length, NaN=True)
   
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))
    
    #chart parameters
    assert form in ['jump','drift','oscillation'], "Undefined shift form"
    shift = delta
    if k is None:
        k = abs(delta)/2
    if L_minus is None:
        L_minus = -L_plus
    n = int(n)
    assert n > 0, "n must be strictly positive"
    n_shift = int(n/2)
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"

    FP_minus = np.zeros((nmc,1)); FP_plus = np.zeros((nmc,1))
    RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
    RL1_plus[:] = np.nan; RL1_minus[:] = np.nan
    for b in range(nmc):
        
        if BB_method == 'MABB': 
            boot = bb.resample_MatchedBB(data, block_length, n=n)
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
    
        if form == 'oscillation':
            eta = np.random.uniform(0.02, 0.2)
            boot[n_shift:] = np.sin(eta*np.pi*np.arange(n_shift))*shift + boot[n_shift:]
            pass
        elif form == 'drift':
            power = np.random.uniform(1.5,2)
            boot[n_shift:] = shift/(500)*(np.arange(n_shift)**power) + boot[n_shift:]
            pass
        else: 
            boot[n_shift:] = boot[n_shift:] + shift
            pass
        
        cnt_plus = 0; cp = 0 
        C_plus = np.zeros((n,1)); nan_p = np.zeros((n,1))
        for i in range(1, n):
            if not np.isnan(boot[i]):
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                C_plus[n_shift] = 0
                cp = 0
            elif (np.isnan(boot[i]) and cp < gap):
                C_plus[n_shift] = 0
                C_plus[i] = C_plus[i-1]
                cp += 1; nan_p[i] = 1
            else: 
                C_plus[i] = 0 
                nan_p[i] = 1
            if C_plus[i] > L_plus and i < n_shift + 1 and cnt_plus == 0:
                ind = nan_p[0:i]
                FP_plus[b] = i #-sum(ind)
                cnt_plus += 1
            elif C_plus[i] > L_plus and i > n_shift: 
                ind = nan_p[n_shift:i]
                RL1_plus[b] = i - n_shift #-sum(ind)
                break 
            
        cnt_minus = 0; cm = 0
        C_minus = np.zeros((n,1)); nan_m = np.zeros((n,1))      
        for j in range(1, n):
            if not np.isnan(boot[j]):
                C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                C_minus[n_shift] = 0
                cm = 0
            elif (np.isnan(boot[j]) and cm < gap):
                C_minus[n_shift] = 0
                C_minus[j] = C_minus[j-1]
                cm += 1; nan_m[j] = 1
            else: 
                C_minus[j] = 0 
                nan_m[j] = 1
            if C_minus[j] < L_minus and j <n_shift + 1 and cnt_minus == 0: # first false positive 
                ind = nan_p[0:j]
                FP_minus[b] = j #-sum(ind) 
                cnt_minus += 1
            elif C_minus[j] < L_minus and j > n_shift: 
                ind = nan_m[n_shift:j]
                RL1_minus[b] = j - n_shift #-sum(ind)
                break
            
        if np.isnan(RL1_plus[b]): 
            RL1_plus[b] = n - n_shift
        if np.isnan(RL1_minus[b]): 
            RL1_minus[b] = n - n_shift
            
        if FP_minus[b] == 0: 
            FP_minus[b] = n_shift
        if FP_plus[b] == 0: 
            FP_plus[b] = n_shift
            
           
    if two_sided:
        ARL1 = (1/(np.nanmean(RL1_minus)) + 1/(np.nanmean(RL1_plus)))**(-1) 
        ARL0 = (1/(np.nanmean(FP_minus)) + 1/(np.nanmean(FP_plus)))**(-1)
    else: 
        ARL1 = np.mean(RL1_plus)
        ARL0 = np.nanmean(FP_plus)
        
    return (ARL1, ARL0)

#############################
    
def shifts_montgomery(data, L_plus, L_minus=None, delta=1.5, k=None, nmc=4000, 
              n=2000, two_sided=True, block_length=None, missing_values='omit', 
                 gap=0, BB_method='MBB'):
    """ 
    Estimates the shift sizes of the data with an optimal formula. 
    
    The sizes of the shifts are estimated after each alert using a
    classical formula (Montgomery, Introduction to statistical 
    quality control, 2004) on the out-of-control (OC) series.
     
    Parameters
    ---------
    data : 2D-array
        OC dataset (rows: time, columns: OC series).
    L_plus : float 
        Value for the positive control limit.
    L_minus : float, optional
        Value for the negative control limit. Default is None. 
        When None, L_minus = - L_plus. 
    delta : float, optional
        The target shift size. Default is 1.5.
    k : float, optional
        The allowance parameter.  The default is None.
        When None, k = delta/2 (optimal formula for iid normal data).
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    two_sided : bool, optional
        Flag to use two-sided CUSUM chart. Otherwise, the one-sided 
        upper CUSUM chart is used. Default is True.
    block_length :  int > 0, optional
        The length of the blocks. Default is None. 
        When None, the length is computed using an optimal formula. 
    missing_values : str, optional
        String that indicates how to deal with the missing values (MV). 
        The string value should be chosen among: 'omit', 'reset' and 'fill':
        'omit' removes the blocks containing MV ;
        'fill' fills-up the MV by the mean of each series ;
        'reset' resets the chart statistics at zero for gaps larger than
        a specified gap length (argument 'gap'). 
        The chart statistics is simply propagated through smaller gaps. 
        Default is 'omit'.
    gap :  int >= 0, optional
        The length of the gaps above which the chart statistics are reset,
        expressed in number of obs. Default is zero. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
       
    Returns
    --------
    shifts : 1D-array
        The estimated shift sizes.
        
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    
    if k is None:
        k = abs(delta)/2
    if L_minus is None:
        L_minus = -L_plus
    (n_obs, n_series) = data.shape 
    
    assert missing_values in ['fill', 'reset', 'omit'], "Undefined value for 'missing_values'"
    if missing_values == 'fill':
        for i in range(n_series):
            data[np.isnan(data[:,i]),i] = np.nanmean(data[:,i]) #fill obs by the mean of the series
            
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if missing_values == 'fill' or missing_values == 'omit':
        if BB_method == 'MBB': 
            blocks = bb.MBB(data, block_length)    
        elif BB_method == 'NBB':
            blocks = bb.NBB(data, block_length) 
        elif BB_method == 'CBB':
            blocks = bb.CBB(data, block_length) 
        
    else:
        if BB_method == 'MBB': 
            blocks = bb.MBB(data, block_length, NaN=True)    
        elif BB_method == 'NBB':
            blocks = bb.NBB(data, block_length, NaN=True) 
        elif BB_method == 'CBB':
            blocks = bb.CBB(data, block_length, NaN=True)
   
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))
        
    n = int(n)
    assert n > 0, "n must be strictly positive"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"

    shift_hat_plus = np.zeros((nmc, 1)); shift_hat_minus = np.zeros((nmc, 1))
    shift_hat_plus[:] = np.nan; shift_hat_minus[:] = np.nan
    for b in range(nmc):

        if BB_method == 'MABB': 
            boot = bb.resample_MatchedBB(data, block_length, n=n)
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
    
        C_plus = np.zeros((n,1)); cp = 0
        for i in range(1, n):
            if not np.isnan(boot[i]):
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                cp = 0
            elif (np.isnan(boot[i]) and cp < gap):
                C_plus[i] = C_plus[i-1]
                cp += 1
            else: 
                C_plus[i] = 0 
            if C_plus[i] > L_plus:
                last_zero = np.where(C_plus[:i] == 0)[0][-1]
                shift_hat_plus[b] = k + C_plus[i]/(i - last_zero)
                break 
            
        C_minus = np.zeros((n,1)); cm = 0     
        for j in range(1, n):
            if not np.isnan(boot[j]):
                C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                cm = 0
            elif (np.isnan(boot[j]) and cm < gap):
                C_minus[j] = C_minus[j-1]
                cm += 1
            else: 
                C_minus[j] = 0 
            if C_minus[j] < L_minus:
                last_zero = np.where(C_minus[:j] == 0)[0][-1]
                shift_hat_minus[b] = -k - C_minus[j]/(j - last_zero)
                break
                        
    if two_sided:
        shifts = np.concatenate((shift_hat_plus[np.where(~np.isnan(shift_hat_plus))],
                                shift_hat_minus[np.where(~np.isnan(shift_hat_minus))]))
    else:
        shifts = shift_hat_plus[np.where(~np.isnan(shift_hat_plus))]
        
    return shifts


def shift_size(data, indexIC, dataIC=None, delta=1.5, k=None, 
                          ARL0_threshold=200, block_length=None, qt=0.5, 
                          accuracy=0.1, L_plus=60, missing_values ='omit', 
                          gap=0, verbose=True, BB_method='MBB'):
    """ 
    Iterative algorithm that computes in turn the control limits of the chart 
    and the target shift size. 
    
    First, the program computes the control limit for an initial value 
    of delta, the target shift size. 
    Then, it estimates the actual shifts on the out-of-control (OC)
    processes using a classical formula (Montgomery, Introduction to statistical 
    quality control, 2004). 
    After that, the control limits are readjusted for the new delta. 
    The procedure is iterated until the shift size converges. 
    
    Parameters
    ---------
    data : 2D-array
        Dataset containing the IC and OC series 
        (rows: time, columns: series).
    indexIC : 1D-array          
         Array containing the index of the IC processes among all series.
    dataIC : 2D-array, optional
         IC dataset. Default is None. 
         If set to None, the IC dataset is retrieved from the 'data'.
    delta : float, optional
        An initial value for the shift size. Default is 1.5.
    k : float, optional
        The allowance parameter.  The default is None.
        When None, k = delta/2 (optimal formula for iid normal data).
    ARL0_threshold : int > 0, optional
        Pre-specified value for the IC average run length (ARL0). 
        This value is inversely proportional to the rate of false positives.
        Typical values are 100, 200 or 500. Default is 200.
    block_length :  int > 0, optional
        The length of the blocks. Default is None. 
        When None, the length is computed using an optimal formula. 
    qt : float in [0,1], optional
        Quantile of the shift size distribution (used to select an appropriate
        shift size). Default is 0.5.
    accuracy : float > 0, optional
        Accuracy to reach the pre-specified value for ARL0: 
        the algorithm stops when |ARL0-ARL0_threshold|< accuracy.
        The default is 0.1.
    L_plus : float , optional
        Upper value for the positive control limit. Default is 60.
    missing_values : str, optional
        String that indicates how to deal with the missing values (MV). 
        The string value should be chosen among: 'omit', 'reset' and 'fill':
        'omit' removes the blocks containing MV ;
        'fill' fills-up the MV by the mean of each series ;
        'reset' resets the chart statistics at zero for gaps larger than
        a specified gap length (argument 'gap'). 
        The chart statistics is simply propagated through smaller gaps. 
        Default is 'omit'.
    gap :  int >= 0, optional
        The length of the gaps above which the chart statistics are reset,
        expressed in number of obs. Default is zero. 
    Verbose : bool, optional
        Flag to print intermediate results. Default is True.
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
       
    Returns
    ------
    L : float                
        The control limit of the chart.
    delta : float
        The target shift size.
        
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape 
    if dataIC is None:
        dataIC = data[:,indexIC]
    indexOC = np.in1d(np.arange(n_series), indexIC)
    dataOC = data[:,~indexOC]
    delta_prev = 0
    
    assert accuracy > 0, "accuracy must be strictly positive"
    
    while (np.abs(delta - delta_prev) > accuracy):
        L = limit_CUSUM(dataIC, delta=delta, k=k, L_plus=L_plus, ARL0_threshold=ARL0_threshold, 
                            block_length=block_length, missing_values=missing_values,
                            gap=gap, verbose=verbose, BB_method=BB_method)
        delta_prev = delta
        shifts = shifts_montgomery(dataOC, L_plus=L, delta=delta, k=k, 
                          missing_values=missing_values, gap=gap, 
                          block_length=block_length, BB_method=BB_method) 
        delta = np.quantile(np.abs(shifts), qt) 
        if verbose:
            print('control limit: ',  L)
            print('shift size: ', delta)
        
    return (L, delta)

##########################################################################"
""" Tests """

if __name__ == "__main__":            

    
    #### iid normal data tests ####
    data = np.random.normal(0 ,1, size=(10000,1))
    #paper 'basics about the CUSUM'
    t1 = ARL0_CUSUM(data, delta=1, L_plus=4, block_length=1) #168
    t1a = ARL0_CUSUM(data, delta=1, L_plus=4, block_length=1, BB_method='CBB') #168
    t1b = ARL0_CUSUM(data, delta=1, L_plus=4, block_length=1, BB_method='NBB') #168
    t1c = ARL0_CUSUM(data, delta=1, L_plus=4, block_length=1, BB_method='MABB') #168
    t2 = ARL0_CUSUM(data, delta=1, L_plus=5, block_length=1) #465
    
    t3 = limit_CUSUM(data, delta=1, ARL0_threshold=465, verbose=True, block_length=1)#5
    t3a = limit_CUSUM(data, delta=1, ARL0_threshold=465, block_length=1, BB_method='CBB')#5
    t3b = limit_CUSUM(data, delta=1, ARL0_threshold=465, block_length=1, BB_method='NBB')#5
    t3c = limit_CUSUM(data, delta=1, ARL0_threshold=465, block_length=1, BB_method='MABB')#5
    
    t4 = ARL1_CUSUM(data, delta=2, L_plus=4, k=1/2, block_length=1) #3.34
    t4a = ARL1_CUSUM(data, delta=2, L_plus=4, k=1/2, block_length=1, BB_method='CBB') #3.34
    t4b = ARL1_CUSUM(data, delta=2, L_plus=4, k=1/2, block_length=1, BB_method='NBB') #3.34
    t4c = ARL1_CUSUM(data, delta=2, L_plus=4, k=1/2, block_length=1, BB_method='MABB') #3.34
    t5 = ARL1_CUSUM(data, delta=2, L_plus=5, k=1/2, block_length=1) #4.01
    #Qiu example 4.3
    t6 = ARL1_CUSUM(data, delta=0.25, L_plus=3.502, k=1/2, block_length=1) #52.836
    
    t7 = ARL_values(data, delta=2, L_plus=4, k=1/2, block_length=1) #3.34-168
    t7a = ARL_values(data, delta=2, L_plus=4, k=1/2, block_length=1, BB_method='CBB') #3.34-168
    t7c = ARL_values(data, delta=2, L_plus=4, k=1/2, block_length=1, BB_method='MABB') #3.34-168
    t8 = ARL_values(data, delta=2, L_plus=5, k=1/2, block_length=1) #4.01-465
    
    #####################"
    
    n = 1000
    delta = 2
    dataIC = np.random.normal(0, 1, size=(n,4))
    dataOC = np.random.normal(delta, 1, size=(n,4))
    data = np.hstack((dataIC, dataOC))
    indexIC = np.arange(0, 4)
    
    L1 = limit_CUSUM(dataIC, L_plus=5, delta=1.5, block_length=1) #2.9-3
    shifts = shifts_montgomery(data, L1, block_length=1) 
    d1 = np.quantile(shifts, 0.5) #1.8-2
    L2, d2 = shift_size(data=data, indexIC=indexIC, delta=delta, 
                          block_length=1, L_plus=5) #delta=2.4-2.3 and L=1.8-1.9
