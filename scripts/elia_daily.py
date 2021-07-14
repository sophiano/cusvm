# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:43:54 2021

@author: sopmathieu

This script applies the cusvm procedure to the load factors of the photovoltaic 
energy production in Belgium.
The data are obtained each day and cover the years 2015-2020.

"""

import pickle
import pandas as pd
import numpy as np 
from datetime import timedelta, datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
import sys
sys.path.insert(1, '../')

from cusvm import preprocessing as pre
from cusvm import autocorrelations as acf
from cusvm import cusum_design_bb as chart
from cusvm import svm_training as svm
from cusvm import alerts as appl

### load data 
df = pd.read_csv (r'../data/PVdaily.csv')
data = np.array(df)[:,1:]
data = data.astype('float')
data = data/96
names = list(df.columns)[1:]
(n_obs, n_series) = data.shape   


### prepare the time
def is_leap_year(year):
    """Determine whether a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def toYearFraction(date):
    """ convert a date into a fraction of year """
    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)

    yearElapsed = len([startOfThisYear + timedelta(days=0, minutes=x) for x in range(0, int((date-startOfThisYear).total_seconds() / 60), 15)])
    yearDuration = len([startOfThisYear + timedelta(days=0, minutes=x) for x in range(0, int((startOfNextYear-startOfThisYear).total_seconds() / 60), 15)])
    fraction = yearElapsed/yearDuration

    return date.year + fraction


# date = df['Date']
# n_obs = len(date)

# date_obj = []
# time = []
# for i in range(n_obs):
#     date_obj.append(datetime.strptime(date[i], '%Y-%m-%d'))
#     time.append(toYearFraction(date_obj[i]))

# time = np.array(time)

# export data (time in fraction of years)
# with open('../data/time_daily', 'wb') as file: 
#       my_pickler = pickle.Pickler(file)
#       my_pickler.dump(time) 


### load time
with open('../data/time_daily', 'rb') as file:
    my_depickler = pickle.Unpickler(file)
    time = my_depickler.load() #data every day
    
    
### remove year 2015 which contains unusual deviations
# start = np.where(time >= 2016)[0][0] 
# data = data[start:,:]
# time = time[start:]

### plot the data 
def plot(data, time, start_time=2015, stop_time=2021, ind=[3,4,14,20,21], same_ax=False):
    
    start = np.where(time >= start_time)[0][0] 
    stop = np.where(time >= stop_time)[0][0] 
    if stop-start < 1000:
        ind_ticks = np.arange(start, stop, 60) #two months
        x_ticks = np.round(time[ind_ticks],2)
    else :
        ind_ticks = np.arange(start, stop, 365) #one year
        x_ticks = np.round(time[ind_ticks])
        
        
    count = 1
    fig = plt.figure(figsize=(10.0, 12.0))
    max_val = np.max(data[start:stop, ind])*1.1
    for i in ind:
        f = fig.add_subplot(len(ind), 1, count)
        plt.ylabel(names[i])
        #plt.plot(data[start:stop, i])
        plt.plot(time[start:stop], data[start:stop, i])
        if same_ax:
            f.set_ylim([0, max_val])
        if count < len(ind):
            f.axes.get_xaxis().set_ticklabels([]) 
        plt.xticks(x_ticks)
        count += 1
        
    plt.show()

plot(data, time, 2016, 2017) #RESA, IVEG, IMEA 2015
plot(data, time, 2019, 2020) #14 September, IMEA (pic)
plot(data, time, 2017, 2017.8) #8march-21May (par 1/4H, changements brusques, avec 0)

plot(data, time, 2016, 2016+6/12) #RESA, IVEG, IMEA 2015
plot(data, time)

#=====================================================================
### Preprocessing
#======================================================================


### additive or multiplicative model 
ts = pd.Series(data[:,11], time)
# decomposition = sm.tsa.seasonal_decompose(ts, model='additive', period=365)
# decomposition.plot()
decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative', period=365)
decomposition.plot()
ts = pd.Series(data[:,7], time)
decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative', period=365)
decomposition.plot()

### rescaling
data_rescaled, k_factors = pre.rescaling(data, period_rescaling=365) 
plot(data_rescaled, time, 2016)

### median
med = pre.median(data_rescaled)
fig = plt.figure(figsize=(10.0, 6.0))
plt.plot(time, med) ; plt.show()

### remove common signal 
ratio = pre.remove_signal(data, ref=med) #mean = 1
plot(ratio, time)

ratio, k_factors = pre.rescaling(ratio, period_rescaling=365) 
plot(ratio, time)

k_tot = np.round(np.mean(k_factors, axis=0),2)
k_class =  np.argsort(k_tot)
k_class_names = [names[i] for i in k_class]

# sub = pre.remove_signal(data, 'additive', ref=med) #mean = 0
# plot(sub, time)

### remove intrinsic level
# ratio_ws = pre.level_removal(ratio, wdw=365) #1 year
# plot(ratio_ws, time)

### select an IC pool
pool = pre.pool_clustering(ratio) #9 ; 10 if 2016 (IVEG ou IMEA)
names_IC = [names[i] for i in range(n_series) if i in pool]
names_OC = [names[i] for i in range(n_series) if i not in pool]

plot(ratio, time, ind=[2,7,12,14,17], same_ax=True) #OC
plot(ratio, time, ind=[9,11,18,19,21], same_ax=True) #IC
plot(ratio, time, ind=[1,21,2,8,15], same_ax=True) #IC and OC

ratioIC = ratio[:, pool]

### standardise the data
#K_knee = pre.choice_K(ratio, ratioIC, start=50, stop=2000, step=50)
K = 400 
data_stn, dataIC_stn = pre.standardisation(ratio, ratioIC, K)
plot(data_stn, time)

### autocorrelation
acf.acf_pacf_plot(data_stn, which_display=3, max_cov=100) 
acf.acf_pacf_plot(data_stn, which_display=5, max_cov=100) 
acf.acf_pacf_plot(data_stn, which_display=21, max_cov=100) 


#=========================================================================
### design of the chart 
#=========================================================================

### choice of the block length 
#bb_length = acf.block_length_choice(dataIC_stn, 1, 50, 1) 
bb_length = 8 

delta_init = 2 #intial value for the target shift size
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)

### adjust the control limits
# control_limit, delta_target = chart.shift_size(data_stn, pool, L_plus=10,
#                 delta=delta_init, ARL0_threshold=ARL0, block_length=bb_length, 
#                 qt=0.4, missing_values ='omit') #3.2 - 2.24 

delta_target = 2

# control_limit = chart.limit_CUSUM(dataIC_stn, delta=delta_target, L_plus=4,
#                 block_length=bb_length, missing_values='omit') #2.4 for delta=3

control_limit = 3.3

#=========================================================================
### train support vector classifier and regressor 
#=========================================================================

### select the length of the input vector 
wdw_length = svm.input_vector_length(dataIC_stn, delta_target, control_limit,
                                block_length=bb_length, qt=0.95) #7

wdw_length = 10 


### train and validate the models
start = np.where(time >= 2015)[0][0]
stop = np.where(time >= 2016)[0][0]
std_data = np.nanstd(data_stn[start:stop,:]) #2.5
scale = 4 #scale parameter (~variance) of the halfnormal distribution
n = 21000*3 #number of testing and training instances
n_search = 12000*3 #number of testing and training instances

### find an optimal value for C (regularization parameter)
# C_choices = svm.choice_C(dataIC_stn, control_limit, delta_target, wdw_length, scale,
#                     start = 1, stop = 11, step = 1,
#               delay=True, n=n_search, block_length=bb_length, confusion=False)

#C = C_choices[2] #4,4,2

C = 4

### train the classifier and regressor with selected C and kernel
reg, clf = svm.training_svm(dataIC_stn, control_limit, delta_target,
                wdw_length, scale, delay=True, n=n, C=C, block_length=bb_length)

### save models 
# filename = 'svr_elia_daily.sav'
# pickle.dump(reg, open(filename, 'wb'))
# filename = 'svc_elia_daily.sav'
# pickle.dump(clf, open(filename, 'wb'))

### or load the models previously trained
reg = pickle.load(open('../svm_models/svr_elia_daily.sav', 'rb'))
clf = pickle.load(open('../svm_models/svc_elia_daily.sav', 'rb'))

#=========================================================================
### run the control chart and plot results (with svm predictions)
#=========================================================================

region = [i for i in range(len(names)) if names[i] == 'Resa'][0]

for i in range(region, region+1): 
#for i in range(n_series): 
    
    data_indv = data_stn[:,i] #monitored series
    
    [form_plus, form_minus, size_plus, size_minus,
    C_plus, C_minus] = appl.alerts_info(data_indv, control_limit, 
            delta_target, wdw_length, clf, reg)
                                  
    fig = appl.plot_3panels(data_indv, control_limit, time, 
                      form_plus, form_minus, size_plus, size_minus, 
                      C_plus, C_minus, names[i], time_start=2015,
                      time_stop=2021)
    
    # fig = appl.plot_4panels(data_indv, data[:,i], control_limit, time, 
    #              form_plus, form_minus, size_plus, size_minus, 
    #              C_plus, C_minus, names[i], time_start=2015,
    #              time_stop=2021)
        
    # fig = appl.plot_1panel(data_indv, time, 
    #          form_plus, form_minus, size_plus, size_minus, names[i],
    #          time_start=2016, time_stop=2017)
        
    ##fig.savefig('../figures/%s.pdf' %names[i]) 
    
    

