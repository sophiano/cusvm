# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:58:08 2020

@author: sopmathieu

This file contains different functions related to the block bootstrap 
procedures. 

These functions can transform an initial panel of time series into 
blocks of consecutive observations. They can also draw a sample of a desired 
length from the data by using different block bootstrap methods.
The functions can accomodate the missing values of the series. 

Four different block bootstrap methods are implemented here :
    - moving block bootstrap (MBB)
    - non-overlapping block bootstrap (NBB)
    - circular block bootstrap (CBB)
    - matched block bootstrap (MABB).

"""

import numpy as np    
import scipy.stats as ss
from statsmodels.tsa.arima_process import ArmaProcess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample


def MBB(x, block_length=None, NaN=False, all_NaN=True):
    """
    Transforms an initial panel of time-series with potentially missing 
    observations into blocks (useful for upcoming moving block bootstrap (MBB)   
    procedures).
    
    Parameters
    ---------
    x : 2D_array
        The dataset (row= obs, col= series) to be decomposed into blocks.
    block_length : int>0 or None
        The length of the blocks. Default is None.
        When None, the length is calculated using an optimal formula.
    NaN : bool, optional
        Flag to keep the blocks containing NaNs (default is False).
        Otherwise blocks with NaNs are removed from the output. 
    all_NaN : bool, optional
        Flag to remove the blocks containing only NaNs. 
        The default is true. 
    
    Returns
    -------
    blocks : 2D-array
         Matrix where each row corresponds to a block of consecutive obs.
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (row_x, column_x) = x.shape   
    
    if block_length is None:
        block_length = int(np.floor(row_x**(1/3))) 
    
    block_length = int(block_length)
    assert block_length > 0, "The block length should be strictly positive."
    

    N = row_x - block_length + 1
    blocks = np.zeros((N*column_x, block_length))
    c = 0
    for j in range(column_x): 
        for i in range(N):
            blocks[c,:] = x[i:i+block_length,j]
            c += 1
            
    if all_NaN:
        blocks = blocks[~np.isnan(blocks).all(axis=1)] #remove rows with all NaNs
    if not NaN:
        blocks = blocks[~np.isnan(blocks).any(axis=1)] #remove rows with any NaNs
    return blocks


def resample_MBB(x, block_length=None, n=None, NaN=False, all_NaN=True):
    """
    Draws a sample of length n randomly selected (with repetitions) from an initial
    panel of series using the MBB method. 
	    
    Parameters
    ----------
    x : 2D_array
        The dataset (row= obs, col= series) to be sampled by the MBB.
    block_length : int>0 or None
        The length of the blocks (default is None).
        When None, the length is calculated using an optimal formula.
    n : int>0 or None
        The length of the output sample (default is None). 
        When None, the length is defined as the number of rows in x.
    NaN : bool, optional   
        Flag to keep the blocks containing NaNs (default is False).
        Otherwise blocks with NaNs are removed from the output.  
    all_NaN : bool, optional
        Flag to remove the blocks containing only NaNs. 
        The default is true. 
    
    Returns
    -------
    sample: 1D-array
        A sample of length n resampled by MBB.
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (row_x, column_x) = x.shape
    if n is None:
        n = row_x #return a series with same length as the original by default
        
    blocks = MBB(x, block_length, NaN, all_NaN)    
    n_blocks = int(np.ceil(n / blocks.shape[1]))
    sample = resample(blocks, replace=True, n_samples=n_blocks)
    return sample.flatten()[:n]


def NBB(x, block_length=None, NaN=False, all_NaN=True):
    """
    Transforms an initial panel of time-series with potentially missing 
    observations into blocks (useful for upcoming non-overlapping block bootstrap             
    (NBB) procedures).
    
    Parameters
    ---------
    x : 2D_array
        The dataset (row= obs, col= series) to be decomposed into blocks.
    block_length : int>0 or None
        The length of the blocks (default is None).
        When None, the length is calculated using an optimal formula.
    NaN : bool, optional
        Flag to keep the blocks containing NaNs (default is False).
        Otherwise blocks with NaNs are removed from the output. 
    all_NaN : bool, optional
        Flag to remove the blocks containing only NaNs. 
        The default is true. 
    
    Returns
    -------
    blocks : 2D-array
         Matrix where each row corresponds to a block of consecutive obs.
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (row_x, column_x) = x.shape
    
    if block_length is None:
        block_length = int(np.floor(row_x**(1/3)))  
    
    block_length = int(block_length)
    assert block_length > 0, "The block length should be strictly positive."

    N = int(np.floor(row_x / block_length))
    blocks = np.zeros((N*column_x, block_length))
    c = 0
    for j in range(column_x):
        it = 0
        for i in range(0,N):
            blocks[c,:] = x[it:it+block_length,j] #non-overlapping
            it += block_length
            c += 1
            
    if all_NaN:
        blocks = blocks[~np.isnan(blocks).all(axis=1)] #remove rows with all NaNs
    if not NaN:
        blocks = blocks[~np.isnan(blocks).any(axis=1)] #remove NaN
    return blocks

def resample_NBB(x, block_length=None, n=None, NaN=False, all_NaN=True):
    """ 
    Draws a sample of length n randomly selected (with repetitions) from an initial panel 
    of series using the NBB method. 
	    
    Parameters
    ----------
    x : 2D_array
        The dataset (row= obs, col= series) to be sampled by the NBB.
    block_length : int>0 or None
        The length of the blocks (default is None).
        When None, the length is calculated using an optimal formula.
    n : int>0 or None
        The length of the output sample (default is None). 
        When None, the length is defined as the number of rows in x.
    NaN: bool, optional   
        Flag to keep the blocks containing NaNs (default is False).
        Otherwise blocks with NaNs are removed from the output. 
    all_NaN : bool, optional
        Flag to remove the blocks containing only NaNs. 
        The default is true.         
    
    Returns
    -------
    sample: 1D-array
        A sample of length n resampled by NBB.
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (row_x, column_x) = x.shape
    if n is None:
        n = row_x #return a series with same length as the original by default
    blocks = NBB(x, block_length, NaN, all_NaN)    
    n_blocks = int(np.ceil(n / blocks.shape[1]))
    sample = resample(blocks, replace=True, n_samples=n_blocks)
    return sample.flatten()[:n]

def CBB(x, block_length=None, NaN=False, all_NaN=True):
    """
    Transforms an initial panel of time-series with potentially missing 
    observations into blocks (useful for upcoming circular block bootstrap             
    (CBB) procedures).
    
    Parameters
    ---------
    x : 2D_array
        The dataset (row= obs, col= series) to be decomposed into blocks.
    block_length : int>0 or None
        The length of the blocks (default is None).
        When None, the length is calculated using an optimal formula.
    NaN : bool, optional
        Flag to keep the blocks containing NaNs (default is False).
        Otherwise blocks with NaNs are removed from the output. 
    all_NaN : bool, optional
        Flag to remove the blocks containing only NaNs. 
        The default is true. 
    
    Returns
    -------
    blocks : 2D-array
         Matrix where each row corresponds to a block of consecutive obs.
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (row_x, column_x) = x.shape
    
    if block_length is None:
        block_length = int(np.floor(row_x**(1/3))) 
    
    block_length = int(block_length)
    assert block_length > 0, "The block length should be strictly positive."

    N = row_x
    blocks = np.zeros((N*column_x, block_length))
    c = 0
    for j in range(column_x): 
        x_dup = np.concatenate((x[:,j],x[:,j]))
        for i in range(0,N):
            blocks[c,:] = x_dup[i:i+block_length] #overlapping
            c += 1
            
    if all_NaN:
        blocks = blocks[~np.isnan(blocks).all(axis=1)] #remove rows with all NaNs
    if not NaN:
        blocks = blocks[~np.isnan(blocks).any(axis=1)] #remove NaN
    return blocks
   

def resample_CBB(x, block_length=None, n=None, NaN=False, all_NaN=True):
    """
    Draws a sample of length n randomly selected (with repetitions) from an initial panel 
    of series using the CBB method. 
	    
    Parameters
    ----------
    x : 2D_array
        The dataset (row= obs, col= series) to be sampled by the CBB.
    block_length : int>0 or None
        The length of the blocks (default is None).
        When None, the length is calculated using an optimal formula.
    n : int>0 or None
        The length of the output sample (default is None). 
        When None, the length is defined as the number of rows in x.
    NaN: bool, optional   
        Flag to keep the blocks containing NaNs (default is False).
        Otherwise blocks with NaNs are removed from the output.      
    all_NaN : bool, optional
        Flag to remove the blocks containing only NaNs. 
        The default is true. 
    
    Returns
    -------
    sample: 1D-array
        A sample of length n resampled by CBB.
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (row_x, column_x) = x.shape
    if n is None:
        n = row_x #return a series with same length as the original by default
    blocks = CBB(x, block_length, NaN, all_NaN)    
    n_blocks = int(np.ceil(n / blocks.shape[1]))
    sample = resample(blocks, replace=True, n_samples=n_blocks)
    return sample.flatten()[:n]


def resample_MatchedBB(x, block_length=None, k_block=None, n=None):
    """
    Draws a sample of length n randomly selected (with repetitions) from an initial panel 
    of series using the matched block bootstrap method.
    
    Parameters
    ----------
    x : 2D_array
        The dataset (row= obs, col= series) to be sampled by the matched BB.
    block_length : int>0 or None
        The length of the blocks (default is None).
        When None, the length is calculated using an optimal formula.
    k_block : float>0 or None     
        Parameter of the matched BB.
        The new blocks are selected according to the following rule:  
        for a current block i, the next block appended to the series is those 
        whose rank is randomly selected between Ri-k_block and Ri+k_block, 
        where Ri is the rank of the last observations of the current block.
        Default is None. When None, the parameter is choosen using an optimal formula.
    n : int>0 or None
        The length of the output sample (default is None). 
        When None, the length is defined as the number of rows in x.     
    
    Returns
    -------
    boot_matched : 1D-array
        A series of obs. resampled by matched BB, of length n.
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (row_x, column_x) = x.shape
    
    if block_length is None:
        block_length = int(np.floor(row_x**(1/3))) 
    block_length = int(block_length)
    assert block_length > 0, "The block length should be strictly positive."
        
    if k_block is None:
        k_block = 0.84*row_x**(1/5)
        
    if n is None:
        n = row_x #return a series with same length as the original by default
        
    #create non-overlapping blocks
    blocks = NBB(x, block_length) 

    last_values = np.zeros((len(blocks)))
    for i in range(len(blocks)):
        last = blocks[i,-7:]
        if not np.isnan(np.min(last)):
            first_not_nan = np.where(~np.isnan(last))[0][-1] #first non nan value in the last seven values
            last_values[i] = last[first_not_nan]
        else:
            last_values[i] = blocks[i, block_length-1] #nan

    blocks = blocks[~np.isnan(last_values)]#remove blocks whose last values is nan
    last_values = last_values[~np.isnan(last_values)]
    ranks = ss.rankdata(last_values, method='ordinal') - 1 #start at 0 
    sort_ind = np.argsort(ranks)
    n_blocks = int(np.ceil(n / block_length))
    b = len(blocks)
    if k_block > b: #really small number of blocks
        k_block = 1
    
    seed = np.random.randint(b)
    boot_matched = blocks[seed,:] #first block is randomly chosen from all blocks
    rk_boot = ranks[seed]
    for i in range(n_blocks):
        u = int(np.random.uniform(rk_boot-k_block, rk_boot+k_block))
        if u < 0: #beginning
            u = -u
        elif u > b - 1: #end series
            u = 2*b - 1 - u
        loc = sort_ind[u] #location of the block (in original series)
        if loc == b - 1:
            loc = 2*b - 3 - loc
        next_block = blocks[loc+1,:]
        rk_boot = ranks[loc+1]
        boot_matched = np.concatenate((boot_matched, next_block))
        
    return boot_matched[:n]

##########################################################################"
    
""" Tests """

if __name__ == "__main__":
    test = np.arange(198)
    test_matrix = np.transpose(np.tile(test,(4,1)))
    test_MBB = MBB(test_matrix)
    test_NBB = NBB(test_matrix)
    test_CBB = CBB(test_matrix)
    
    t1 = MBB(test_matrix, block_length=None)
    t2 = resample_MBB(test_matrix)
    t3 = NBB(test_matrix, block_length=None)
    t4 = resample_NBB(test_matrix, None)
    t5 = CBB(test_matrix, block_length=None)
    t6 = resample_CBB(test_matrix, None)
    T7= resample_CBB(test_matrix, 3, 604)
    

    #Tests matched block bootstrap     
    n = 5080
    phi1 = -0.9
    ar1 = np.array([1, phi1])
    ma1 = np.array([1])
    AR_object1 = ArmaProcess(ar1, ma1)
    data = AR_object1.generate_sample(nsample=(n,2))
    (n,N_series) = data.shape
    #plt.figure(1)
    #plt.plot(data[4700:,0])
    #plt.scatter(data[0:n-1,0], data[1:,0]) #correspond to ar(-0.9)
    
    ### apply an MA(81) to create long-memory series
    wdw = 81
    for i in range(N_series):
        ts = pd.Series(data[:,i], index=range(n))
        data[:,i] = ts.rolling(wdw, center=True).mean()
    data = data[40:n-40,:]    
    (n, N_series) = data.shape
    plt.figure(1)
    plt.plot(data[4700:,0])
    
    block_length = 30
    boot_matched = resample_MatchedBB(data ,block_length, 5, n)
    boot_sampled = resample_NBB(data, block_length, n) #simple NBB
    
    """Plot the result"""
    stop = min(len(boot_sampled), len(boot_matched))
    start = stop - 10*block_length
    x = np.arange(start, stop)
    
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.title("Matched block bootstrap")
    plt.plot(x, boot_matched[start:stop])
    for i in range(start, stop, block_length):
        plt.axvline(x = i)
    axes = plt.gca()
    
    plt.subplot(2, 1, 2)
    plt.title("Simple Block bootstrap")
    plt.plot(x, boot_sampled[start:stop])
    for i in range(start, stop, block_length):
        plt.axvline(x=i) #vertical lines
    axes = plt.gca()
    plt.tight_layout()
    plt.show()