# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:35:31 2020

@author: sopmathieu

This file contains functions to apply the monitoring procedure (composed of 
a CUSUM chart and support vector classifier and regressor) on a time-series. 
The script also contains functions to graphically display the results of 
this monitoring. 

"""

import numpy as np
import matplotlib.pyplot as plt

from cusvm import svm_training as svm


def firstNonNan(x):
  """ 
  Finds first non NaN value in an array. 
  
  Parameters
  ----------
  x : 1D-array
     Initial array to retrieve its first non NaN value.
     
  Returns
  ------
  ind : int
      Index of first non NaN value of the array.
      
  """
  for ind in range(len(x)):
    if not np.isnan(x[ind]):
      return ind


def alerts_info(data, L_plus, delta, wdw_length, clf, reg, L_minus = None,
                     k=None, cut=None, verbose=True):
    """
    Applies the two-sided CUSUM chart on a series.
    
    This function returns the size and the form ('jumps', 'drifts', 'oscillating 
    shifts') of the shifts after each alert.
    
    Parameters
    ----------
    data : 1D-array
        A single series of standardized observations to be monitored. 
    L_plus : float 
        Value for the positive control limit.
    delta : float >= 0
        The target shift size. 
    wdw_length : int > 0
        The length of the input vector.
    clf : support vector classification model
       The trained classifier.
    reg : support vector regression model
        The trained regressor.
    L_minus :  float, optional
        Value for the negative control limit. Default is None. 
        When None, L_minus = - L_plus. 
    k : float, optional
        The allowance parameter. The default is None. 
        When None, k = delta/2 (optimal formula for iid normal data).
    cut : float, optional 
        Upper value for the chart statistics. Values of the positive (resp. 
        negative) chart statistics are constrained to be equal to or 
        lower than 'cut' (resp. equal to or superior than -'cut').
        When None, cut is equal to '2L_plus'. The default is None.
    Verbose : bool, optional
        Flag to print the percentage of alert in the series. Default is True.
    
    Returns
    -------
    form_plus : 1D-array
        Predicted shift forms after positive alerts. 
        When no alerts is detected, the shift forms are set to NaNs.
    form_minus : 1D-array
        Predicted shift forms after negative alerts.
        When no alerts is detected, the shift forms are set to NaNs.
    size_plus : 1D-array
        Predicted shift sizes after positive alerts.
        When no alerts is detected, the shift sizes are set to NaNs.
    size_minus : 1D-array
        Predicted shift sizes after negative alerts.
        When no alerts is detected, the shift sizes are set to NaNs.
    C_plus :  
        Values of the positive chart statistic.
    C_minus :  
        Values of the negative chart statistic.
    
    """
    assert np.ndim(data) == 1, "Input data must be a 1D array (one series)"
    n_obs = len(data)
    if L_minus is None:
        L_minus = -L_plus
    if k is None:
        k = abs(delta)/2
    if cut is None:
        cut = L_plus * 2
    
    length = len(data[~np.isnan(data)])
    
    input_minus = np.zeros((n_obs, wdw_length)); input_plus = np.zeros((n_obs, wdw_length))
    input_minus[:] = np.nan; input_plus[:] = np.nan
    flag_plus = np.zeros((n_obs)); flag_minus = np.zeros((n_obs))
    C_plus = np.zeros((n_obs)); C_minus = np.zeros((n_obs))
    
    for i in range(wdw_length, n_obs):
        
        ## CUSUM monitoring
        C_plus[i] =  min(cut, max(0, C_plus[i-1] + data[i] - k))#avoid cusum "explosion"
        if C_plus[i] > L_plus: #alert
            flag_plus[i] = 1
            input_plus[i,:] = data[i+1-wdw_length:i+1]
            
        C_minus[i] = max(-cut, min(0, C_minus[i-1] + data[i] + k))
        if C_minus[i] < L_minus: #alert
            flag_minus[i] = 1
            input_minus[i,:] = data[i+1-wdw_length:i+1]
                            
   
    ## compute percentage of alerts
    oc_p = np.nonzero(flag_plus)
    oc_m = np.nonzero(flag_minus)
    #if alert both for pos and neg limits, count for only one alert
    oc_both = len(set(np.concatenate((oc_p[0], oc_m[0]))))
    #OC_perc = oc_both*100/n_obs #total period
    OC_perc = oc_both*100/length #observing period

    if verbose : 
        print("Percentage of alerts: %0.2f" %OC_perc) 
        
    ## interpolate NaNs in input vectors
    input_minus_valid, ind_minus = svm.fill_nan(input_minus)
    input_plus_valid, ind_plus = svm.fill_nan(input_plus)
    
    ##apply classifier and regressor on (filled-up) input vectors
    size_minus = np.zeros((n_obs)); size_plus = np.zeros((n_obs))
    size_minus[:] = np.nan; size_plus[:] = np.nan
    form_minus = np.zeros((n_obs)); form_plus = np.zeros((n_obs))
    form_minus[:] = np.nan; form_plus[:] = np.nan
    
    if len(ind_minus)>0: #at least one value
        size_minus[ind_minus] = reg.predict(input_minus_valid)
        form_minus[ind_minus] = clf.predict(input_minus_valid)
    if len(ind_plus)>0:
        size_plus[ind_plus] = reg.predict(input_plus_valid)
        form_plus[ind_plus] = clf.predict(input_plus_valid)
    
    return (form_plus, form_minus, size_plus, size_minus, C_plus, C_minus)


#==================================================================
#==================================================================
    
def plot_4panels(data, unstn_data, L_plus, time, form_plus, form_minus, size_plus, 
                      size_minus, C_plus, C_minus, name, L_minus=None, 
                      time_start=None, time_stop=None):
    """
    Plots the main results of the monitoring. 
    
    This function plots different quantities that resume the main features 
    of the monitoring into four panels. 
    The first panel represents the unstandadised data,
    the second shows the standardized data, 
    the third panel displays the CUSUM statistics applied to the standardised data
    in square-root scale
    and the last panel shows the predicted shift sizes and forms.
    
    Parameters
    ---------
    data : 1D-array
        A (single) series of standardized observations to be monitored. 
    unstn_data : 1D-array
        The original series (without standardization).
    L_plus : float 
        Value for the positive control limit.
    time : 1D-array
        An array with the time of the observations (same length as data).
    form_plus : 1D-array
        Predicted shift forms after positive alerts (same length as data). 
        When no alerts is detected, the shift forms should be set to NaNs.
    form_minus : 1D-array
        Predicted shift forms after negative alerts (same length as data).
        When no alerts is detected, the shift forms should be set to NaNs.
    size_plus : 1D-array
        Predicted shift sizes after positive alerts (same length as data).
        When no alerts is detected, the shift sizes should be set to NaNs.
    size_minus : 1D-array
        Predicted shift sizes after negative alerts (same length as data).
        When no alerts is detected, the shift sizes should be set to NaNs.
    C_plus : 1D-array
        Values of the positivie chart statistic (same length as data).
    C_minus : 1D-array     
        Values of the negative chart statistic (same length as data).
    name : str
       The index of the series.
    L_minus :  float, optional
        Value for the negative control limit. Default is None. 
        When None, L_minus = - L_plus. 
    time_start : int, optional
       Starting time value of the plot, in years. The plot shows the period 
       [time_start, time_stop]. Default is None. 
       When None, the first non-Nan observation in the series is used. 
    time_stop : int, optional
      Last time value of the plot, in years. The plot shows the period 
      [time_start, time_stop]. Default is None. 
      When None, the series is represented until the last observation.
      
    Returns
    ------
    fig : a matplotlib figure
        The figure (with four panels).
        
    """
    
    assert np.ndim(data) == 1, "Input data must be a 1D array (one series)"
    n_obs = len(data)
    if time_start is None:
        start = firstNonNan(data)
    else: 
        start = np.where(time >= time_start)[0][0]
    if time_stop is None:
        stop = len(time) - 1 
    else: 
        stop = np.where(time >= time_stop)[0][0]
    assert stop > start, "time_stop should be stricly superior to time_start!"
    
    if L_minus is None:
        L_minus = -L_plus
        
    #colors
    colorInd_plus = np.where(~np.isnan(form_plus))[0][:]
    colorInd_minus = np.where(~np.isnan(form_minus))[0][:]
    color_graph_p = np.ones((n_obs))*3; color_graph_m = np.ones((n_obs))*3 
    color_graph_m[colorInd_minus] = form_minus[colorInd_minus]
    color_graph_p[colorInd_plus] = form_plus[colorInd_plus]
    
    #jumps
    jump_p = np.zeros((n_obs)); jump_p[:]=np.nan
    jump_m = np.zeros((n_obs)); jump_m[:]=np.nan
    jump_m[np.where(color_graph_m == 0)[0]] = size_minus[np.where(color_graph_m == 0)[0]]
    jump_p[np.where(color_graph_p == 0)[0]] = size_plus[np.where(color_graph_p == 0)[0]]
    
    #trends
    trend_p = np.zeros((n_obs)); trend_p[:] = np.nan
    trend_m = np.zeros((n_obs)); trend_m[:] = np.nan
    trend_m[np.where(color_graph_m == 1)[0]] = size_minus[np.where(color_graph_m == 1)[0]]
    trend_p[np.where(color_graph_p == 1)[0]] = size_plus[np.where(color_graph_p == 1)[0]]
    
    #oscillating shifts
    oscill_p = np.zeros((n_obs)); oscill_p[:] = np.nan
    oscill_m = np.zeros((n_obs)); oscill_m[:] = np.nan
    oscill_m[np.where(color_graph_m == 2)[0]] = size_minus[np.where(color_graph_m == 2)[0]]
    oscill_p[np.where(color_graph_p == 2)[0]] = size_plus[np.where(color_graph_p == 2)[0]]


    plt.rcParams['figure.figsize'] = (7.0, 10.0)
    plt.rcParams['font.size'] = 14
    if stop-start <500:
        x_ticks = np.arange(np.round(time[start],1), np.round(time[stop],1), 0.2)
    else :
        x_ticks = np.arange(np.round(time[start]), np.round(time[stop])+1, 1)
        
    data_per = data[start:stop]
    max_val = max(data_per[~np.isnan(data_per)])
    min_val = min(data_per[~np.isnan(data_per)])
    y_max = 1.2*max_val ; y_min = 1.2*min_val
    
    fig = plt.figure()
    f1 = fig.add_subplot(4, 1, 1)
    plt.title("Monitoring in %s" %name)
    plt.ylabel('Unstand. residuals')
    plt.plot(time[start:stop], unstn_data[start:stop])
    plt.plot([time[start], time[stop]], [1, 1], 'k-', lw=2)
    f1.axes.get_xaxis().set_ticklabels([]) 
    f1.set_xlim([time[start], time[stop]])
    plt.xticks(x_ticks)
    f1.axes.xaxis.grid(True, linewidth=0.15)

    f2 = fig.add_subplot(4, 1, 2)
    plt.ylabel('Stand. residuals')
    plt.plot(time[start:stop], data[start:stop])
    plt.plot([time[start], time[stop]], [0, 0], 'k-', lw=2)
    f2.set_ylim([y_min, y_max]); f2.set_xlim([time[start], time[stop]])
    plt.xticks(x_ticks)
    f2.axes.get_xaxis().set_ticklabels([]) 
    f2.axes.xaxis.grid(True, linewidth=0.15)
    
    f3 = fig.add_subplot(4, 1, 3)
    plt.ylabel('CUSUM stats')
    plt.plot(time[start:stop], np.sqrt(C_plus[start:stop]), c='tab:red', label='$C^+$')
    plt.plot(time[start:stop], -np.sqrt(-C_minus[start:stop]), '--', c='tab:brown', label='$C^-$')
    #upper control limit value (horizontal line)
    plt.plot([time[start], time[stop]], [np.sqrt(L_plus), np.sqrt(L_plus)], 'k-', lw=2.2)
    #lower control limit value (horizontal line)
    plt.plot([time[start], time[stop]], [-np.sqrt(abs(L_minus)), -np.sqrt(abs(L_minus))], 'k-', lw=2.2)
    plt.legend(loc='upper right', ncol=2, fontsize=10)
    f3.set_ylim([-np.sqrt(2*L_plus)*2, np.sqrt(2*L_plus)*2])
    f3.set_xlim([time[start], time[stop]])
    plt.xticks(x_ticks)
    f3.axes.get_xaxis().set_ticklabels([]) 
    f3.axes.xaxis.grid(True, linewidth=0.15)
    
    sizes = np.array([size_plus, size_minus])
    sizes_per = sizes[:,start:stop]
    if len(sizes_per[~np.isnan(sizes_per)]) > 0:
        max_val = max(sizes_per[~np.isnan(sizes_per)])
        min_val = min(sizes_per[~np.isnan(sizes_per)])
        y_max = 1.2*max_val ; y_min = 1.2*min_val
    
    f4 = fig.add_subplot(4, 1, 4)
    plt.ylabel('Deviations')
    plt.plot(time[start:stop], jump_m[start:stop], '--', c='tab:purple', label='jumps')
    plt.plot(time[start:stop], jump_p[start:stop], '--', c='tab:purple')
    plt.plot(time[start:stop], trend_m[start:stop],  c='tab:green', label='trends')
    plt.plot(time[start:stop], trend_p[start:stop],  c='tab:green')
    plt.plot(time[start:stop], oscill_m[start:stop], ':', c='tab:orange', label='oscill')
    plt.plot(time[start:stop], oscill_p[start:stop], ':', c='tab:orange')
    plt.plot([time[start], time[stop]], [0, 0], 'k-', lw=2)
    plt.legend(loc='lower right', ncol=3, fontsize=10)
    f4.set_ylim([y_min, y_max]); f4.set_xlim([time[start], time[stop]])
    plt.xticks(x_ticks)
    plt.xlabel('year')
    plt.tick_params(axis='x', which='major') 
    f4.axes.xaxis.grid(True, linewidth=0.15)

    #plt.tight_layout() 
    plt.show()
    return fig 

#==================================================================
### other plots
#==================================================================

    
def plot_1panel(data, time, form_plus, form_minus, size_plus, 
             size_minus, name, time_start=None, time_stop=None):
    """
    Plots the predictions (forms and sizes) of the support 
    vector machines (SVM).

    
    Parameters
    ---------
    data : 1D-array
        A (single) series of standardized observations to be monitored. 
    time : 1D-array
        An array with the time of the observations (same length as data).
    form_plus : 1D-array
        Predicted shift forms after positive alerts (same length as data). 
        When no alerts is detected, the shift forms should be set to NaNs.
    form_minus : 1D-array
        Predicted shift forms after negative alerts (same length as data).
        When no alerts is detected, the shift forms should be set to NaNs.
    size_plus : 1D-array
        Predicted shift sizes after positive alerts (same length as data).
        When no alerts is detected, the shift sizes should be set to NaNs.
    size_minus : 1D-array
        Predicted shift sizes after negative alerts (same length as data).
        When no alerts is detected, the shift sizes should be set to NaNs.
    name : str
       The index or the name of the series.
    time_start : int, optional
       Starting time value of the plot, in years. The plot shows the period 
       [time_start, time_stop]. Default is None. 
       When None, the first non-Nan observation in the series is used. 
    time_stop : int, optional
      Last time value of the plot, in years. The plot shows the period 
      [time_start, time_stop]. Default is None. 
      When None, the series is represented until the last observation.
      
    Returns
    ------
    fig : a matplotlib figure
        The figure with the SVM predictions.
        
    """
    
    assert np.ndim(data) == 1, "Input data must be a 1D array (one series)"
    n_obs = len(data)
    if time_start is None:
        start = firstNonNan(data)
    else: 
        start = np.where(time >= time_start)[0][0]
    if time_stop is None:
        stop = len(time) - 1 
    else: 
        stop = np.where(time >= time_stop)[0][0]
    assert stop > start, "time_stop should be stricly superior to time_start!"
    
        
    #colors
    colorInd_plus = np.where(~np.isnan(form_plus))[0][:]
    colorInd_minus = np.where(~np.isnan(form_minus))[0][:]
    color_graph_p = np.ones((n_obs))*3; color_graph_m = np.ones((n_obs))*3 
    color_graph_m[colorInd_minus] = form_minus[colorInd_minus]
    color_graph_p[colorInd_plus] = form_plus[colorInd_plus]
    
    #jumps
    jump_p = np.zeros((n_obs)); jump_p[:]=np.nan
    jump_m = np.zeros((n_obs)); jump_m[:]=np.nan
    jump_m[np.where(color_graph_m == 0)[0]] = size_minus[np.where(color_graph_m == 0)[0]]
    jump_p[np.where(color_graph_p == 0)[0]] = size_plus[np.where(color_graph_p == 0)[0]]
    
    #trends
    trend_p = np.zeros((n_obs)); trend_p[:] = np.nan
    trend_m = np.zeros((n_obs)); trend_m[:] = np.nan
    trend_m[np.where(color_graph_m == 1)[0]] = size_minus[np.where(color_graph_m == 1)[0]]
    trend_p[np.where(color_graph_p == 1)[0]] = size_plus[np.where(color_graph_p == 1)[0]]
    
    #oscillating shifts
    oscill_p = np.zeros((n_obs)); oscill_p[:] = np.nan
    oscill_m = np.zeros((n_obs)); oscill_m[:] = np.nan
    oscill_m[np.where(color_graph_m == 2)[0]] = size_minus[np.where(color_graph_m == 2)[0]]
    oscill_p[np.where(color_graph_p == 2)[0]] = size_plus[np.where(color_graph_p == 2)[0]]


    plt.rcParams['figure.figsize'] = (12.0, 4.0)
    plt.rcParams['font.size'] = 14
    if stop-start <500:
        x_ticks = np.arange(np.round(time[start], 2), np.round(time[stop], 2), 0.2)
    else :
        x_ticks = np.arange(np.round(time[start]), np.round(time[stop])+1, 1)
    
    
    sizes = np.array([size_plus, size_minus])
    sizes_per = sizes[:,start:stop]
    if len(sizes_per[~np.isnan(sizes_per)]) > 0:
        max_val = max(sizes_per[~np.isnan(sizes_per)])
        min_val = min(sizes_per[~np.isnan(sizes_per)])
        y_max = 1.2*max_val ; y_min = 1.2*min_val
    else:
        data_per = data[start:stop]
        max_val = max(data_per[~np.isnan(data_per)])
        min_val = min(data_per[~np.isnan(data_per)])
        y_max = 1.2*max_val ; y_min = 1.2*min_val
    
    fig = plt.figure()
    f1 = fig.add_subplot(1, 1, 1)
    plt.ylabel('Deviations')
    plt.title("Monitoring in %s" %name)
    plt.plot(time[start:stop], jump_m[start:stop], '--', c='tab:purple', label='jumps')
    plt.plot(time[start:stop], jump_p[start:stop], '--', c='tab:purple')
    plt.plot(time[start:stop], trend_m[start:stop],  c='tab:green', label='trends')
    plt.plot(time[start:stop], trend_p[start:stop],  c='tab:green')
    plt.plot(time[start:stop], oscill_m[start:stop], ':', c='tab:orange', label='oscill')
    plt.plot(time[start:stop], oscill_p[start:stop], ':', c='tab:orange')
    plt.plot([time[start], time[stop]], [0, 0], 'k-', lw=2)
    plt.legend(loc='lower center', ncol=3)
    f1.set_ylim([y_min, y_max]); f1.set_xlim([time[start], time[stop]])
    plt.xticks(x_ticks)
    plt.xlabel('year')
    plt.tick_params(axis='x', which='major') 
    f1.axes.xaxis.grid(True, linewidth=0.15)

    #plt.tight_layout() 
    plt.show()
    return fig 


    
def plot_3panels(data, L_plus, time, form_plus, form_minus, size_plus, 
                      size_minus, C_plus, C_minus, name, L_minus=None, 
                      time_start=None, time_stop=None, years=True, hours=False):
    """
    Plots the main results of the monitoring. 
    
    This function plots different quantities that resume the main features 
    of the monitoring into three panels. 
    The first panel represents the standardized data 
    the second panel displays the CUSUM statistics applied to the standardised 
    data in square-root scale
    and the last panel shows the predicted shift sizes and forms.
    
    Parameters
    ---------
    data : 1D-array
        A (single) series of standardized observations to be monitored. 
    L_plus : float 
        Value for the positive control limit.
    time : 1D-array
        An array with the time of the observations (same length as data).
    form_plus : 1D-array
        Predicted shift forms after positive alerts (same length as data). 
        When no alerts is detected, the shift forms should be set to NaNs.
    form_minus : 1D-array
        Predicted shift forms after negative alerts (same length as data).
        When no alerts is detected, the shift forms should be set to NaNs.
    size_plus : 1D-array
        Predicted shift sizes after positive alerts (same length as data).
        When no alerts is detected, the shift sizes should be set to NaNs.
    size_minus : 1D-array
        Predicted shift sizes after negative alerts (same length as data).
        When no alerts is detected, the shift sizes should be set to NaNs.
    C_plus : 1D-array
        Values of the positivie chart statistic (same length as data).
    C_minus : 1D-array     
        Values of the negative chart statistic (same length as data).
    name : str
       The index or the name of the series.
    L_minus :  float, optional
        Value for the negative control limit. Default is None. 
        When None, L_minus = - L_plus. 
    time_start : int, optional
       Starting time value of the plot. The plot shows the period 
       [time_start, time_stop]. Default is None. 
       When None, the first non-Nan observation in the series is used. 
    time_stop : int, optional
      Last time value of the plot. The plot shows the period 
      [time_start, time_stop]. Default is None. 
      When None, the series is represented until the last observation.
    years : bool, optional
        Flag to indicate that the time of the obs. is expressed into years.
        The default is True.
    hours : bool, optional
        Flag to indicate that the time of the obs. is expressed into minutes or
        hours. The function is then designed to show a single day of data 
        (xthick each two hours).
        The default is False.
    Returns
    ------
    fig : a matplotlib figure
        The figure (with four panels).
        
    """
    
    assert np.ndim(data) == 1, "Input data must be a 1D array (one series)"
    n_obs = len(data)
    if time_start is None:
        start = firstNonNan(data)
    else: 
        start = np.where(time >= time_start)[0][0]
    if time_stop is None:
        stop = len(time) - 1 
    else: 
        stop = np.where(time >= time_stop)[0][0]
    assert stop > start, "time_stop should be stricly superior to time_start!"
    
    if L_minus is None:
        L_minus = -L_plus
        
    #colors
    colorInd_plus = np.where(~np.isnan(form_plus))[0][:]
    colorInd_minus = np.where(~np.isnan(form_minus))[0][:]
    color_graph_p = np.ones((n_obs))*3; color_graph_m = np.ones((n_obs))*3 
    color_graph_m[colorInd_minus] = form_minus[colorInd_minus]
    color_graph_p[colorInd_plus] = form_plus[colorInd_plus]
    
    #jumps
    jump_p = np.zeros((n_obs)); jump_p[:]=np.nan
    jump_m = np.zeros((n_obs)); jump_m[:]=np.nan
    jump_m[np.where(color_graph_m == 0)[0]] = size_minus[np.where(color_graph_m == 0)[0]]
    jump_p[np.where(color_graph_p == 0)[0]] = size_plus[np.where(color_graph_p == 0)[0]]
    
    #trends
    trend_p = np.zeros((n_obs)); trend_p[:] = np.nan
    trend_m = np.zeros((n_obs)); trend_m[:] = np.nan
    trend_m[np.where(color_graph_m == 1)[0]] = size_minus[np.where(color_graph_m == 1)[0]]
    trend_p[np.where(color_graph_p == 1)[0]] = size_plus[np.where(color_graph_p == 1)[0]]
    
    #oscillating shifts
    oscill_p = np.zeros((n_obs)); oscill_p[:] = np.nan
    oscill_m = np.zeros((n_obs)); oscill_m[:] = np.nan
    oscill_m[np.where(color_graph_m == 2)[0]] = size_minus[np.where(color_graph_m == 2)[0]]
    oscill_p[np.where(color_graph_p == 2)[0]] = size_plus[np.where(color_graph_p == 2)[0]]


    plt.rcParams['figure.figsize'] = (7.0, 7.0)
    plt.rcParams['font.size'] = 14
    if hours:
        x_ticks = np.arange(time[start], time[stop], (time[stop]-time[start])/12)
    elif years:
        x_ticks = np.arange(np.round(time[start]), np.round(time[stop])+1, 1)
    else :
         x_ticks = np.arange(np.round(time[start],1), np.round(time[stop],1), 0.2)
        
    data_per = data[start:stop]
    max_val = max(data_per[~np.isnan(data_per)])
    min_val = min(data_per[~np.isnan(data_per)])
    y_max = 1.2*max_val ; y_min = 1.2*min_val
    
    fig = plt.figure()
    f1 = fig.add_subplot(3, 1, 1)
    plt.title("Monitoring in %s" %name)
    plt.ylabel('Stand. residuals')
    plt.plot(time[start:stop], data[start:stop])
    plt.plot([time[start], time[stop]], [0, 0], 'k-', lw=2)
    f1.set_ylim([y_min, y_max]); f1.set_xlim([time[start], time[stop]])
    plt.xticks(x_ticks)
    f1.axes.get_xaxis().set_ticklabels([]) 
    f1.axes.xaxis.grid(True, linewidth=0.1, linestyle='-', color='#7f7f7f')
    
    f2 = fig.add_subplot(3, 1, 2)
    plt.ylabel('CUSUM stats')
    plt.plot(time[start:stop], np.sqrt(C_plus[start:stop]), c='tab:red', label='$C^+$')
    plt.plot(time[start:stop], -np.sqrt(-C_minus[start:stop]), '--', c='tab:brown', label='$C^-$')
    #upper control limit value (horizontal line)
    plt.plot([time[start], time[stop]], [np.sqrt(L_plus), np.sqrt(L_plus)], 'k-', lw=2.2)
    #lower control limit value (horizontal line)
    plt.plot([time[start], time[stop]], [-np.sqrt(abs(L_minus)), -np.sqrt(abs(L_minus))], 'k-', lw=2.2)
    plt.legend(loc='upper right', ncol=2, fontsize=10)
    f2.set_ylim([-np.sqrt(2*L_plus)*2, np.sqrt(2*L_plus)*2])
    f2.set_xlim([time[start], time[stop]])
    plt.xticks(x_ticks)
    f2.axes.get_xaxis().set_ticklabels([]) 
    f2.axes.xaxis.grid(True, linewidth=0.1, linestyle='-', color='#7f7f7f')
    
    sizes = np.array([size_plus, size_minus])
    sizes_per = sizes[:,start:stop]
    if len(sizes_per[~np.isnan(sizes_per)]) > 0:
        max_val = max(sizes_per[~np.isnan(sizes_per)])
        min_val = min(sizes_per[~np.isnan(sizes_per)])
        y_max = 1.2*max_val ; y_min = 1.2*min_val

    
    f3 = fig.add_subplot(3, 1, 3)
    plt.ylabel('Deviations')
    plt.plot(time[start:stop], jump_m[start:stop], '--', c='tab:purple', label='jumps')
    plt.plot(time[start:stop], jump_p[start:stop], '--', c='tab:purple')
    plt.plot(time[start:stop], trend_m[start:stop],  c='tab:green', label='trends')
    plt.plot(time[start:stop], trend_p[start:stop],  c='tab:green')
    plt.plot(time[start:stop], oscill_m[start:stop], ':', c='tab:orange', label='oscill')
    plt.plot(time[start:stop], oscill_p[start:stop], ':', c='tab:orange')
    plt.plot([time[start], time[stop]], [0, 0], 'k-', lw=2)
    plt.legend(loc='lower right', ncol=3, fontsize=10)
    plt.xticks(x_ticks)
    if hours:
        f3.axes.set_xticklabels(list(np.arange(0,24,2))) #thick each two hours
        plt.xlabel('hour')
    else:
        f3.set_ylim([y_min, y_max]) 
        f3.set_xlim([time[start], time[stop]])
        plt.xlabel('year')
    plt.tick_params(axis='x', which='major') 
    f3.axes.xaxis.grid(True, linewidth=0.1, linestyle='-', color='#7f7f7f')

    #plt.tight_layout() 
    plt.show()
    return fig 