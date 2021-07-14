# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:15:47 2021

@author: sopmathieu

This script applies the cusvm procedure to the load factors of the photovoltaic 
energy production in Belgium.
The data are obtained each quarter of an hour and cover the years 2015-2020.
"""

import pickle
import pandas as pd
import numpy as np 
from datetime import timedelta, datetime
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
df = pd.read_csv (r'../data/PVq.csv')
data = np.array(df)[4:,1:] 
data = data.astype('float')
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


# date = np.array(df['dt'][4:]) #[70182:105222]
# date = [date[i].replace('T', '-') for i in range(n_obs)]
# date = [date[i].replace('Z', '') for i in range(n_obs)]
# n_obs = len(date)


# date_obj = datetime.strptime(date[0], '%Y-%m-%d-%H:%M:%S')
# time_start = toYearFraction(date_obj)
# date_obj = datetime.strptime(date[-1], '%Y-%m-%d-%H:%M:%S')
# time_stop = toYearFraction(date_obj)
# time = np.arange(time_start, time_stop, (time_stop-time_start)/n_obs)


## export data (time in fraction of years)
# with open('../data/time_minutes', 'wb') as file: 
#       my_pickler = pickle.Pickler(file)
#       my_pickler.dump(time) 


### load time
with open('../data/time_minutes', 'rb') as file:
    my_depickler = pickle.Unpickler(file)
    time = my_depickler.load() #data every 15 minutes
    
    
### remove year 2015 with unusual deviations
# start = np.where(time >= 2016)[0][0] 
# data = data[start:,:]
# time = time[start:]

### plot the data 
def plot(data, time, start_time=2017, stop_time=2018, ind=[3,4,14,20,21], same_ax=False):
    
    start = np.where(time >= start_time)[0][0] 
    stop = np.where(time >= stop_time)[0][0] 
    if stop-start < 1000:
        ind_ticks = np.arange(start, stop, 30*96) #one month
        x_ticks = np.round(time[ind_ticks],2)
    else :
        ind_ticks = np.arange(start, stop, 180*96) #6 months
        x_ticks = np.round(time[ind_ticks],1)
        
    max_val = max(data[~np.isnan(data)])
    min_val = min(data[~np.isnan(data)])
    y_max = 1.2*max_val
    y_min = 1.2*min_val
    count = 1
    fig = plt.figure(figsize=(10.0, 12.0))
    for i in ind:
        f = fig.add_subplot(len(ind), 1, count)
        plt.ylabel(names[i])
        #plt.plot(data[start:stop, i])
        plt.plot(time[start:stop], data[start:stop, i])
        if same_ax:
            f.set_ylim([y_min,y_max])
        if count < len(ind):
            f.axes.get_xaxis().set_ticklabels([]) 
        plt.xticks(x_ticks)
        count += 1
    plt.show()

plot(data, time, 2015, 2016) 
plot(data, time, 2019, 2020) 

plot(data, time, 2017.2, 2017.5, [1,2,6,7,16])
plot(data, time, 2017, 2018) 
plot(data, time, 2018, 2019) 
plot(data, time, 2016, 2017)

#=====================================================================
### Preprocessing
#======================================================================

### remove zero (nights)
(n_obs, n_series) = data.shape  
ind = np.where(data[:,0]!=0)[0]
data_wht_zero = np.zeros((n_obs, n_series))
data_wht_zero[:] = np.nan
data_wht_zero[ind,:] = data[ind, :]
data = data_wht_zero


### additive or multiplicative model 
# ts = pd.Series(data[:,21], time)
# decomposition = sm.tsa.seasonal_decompose(ts, model='additive', period=96)
# decomposition.plot()
# decomposition = sm.tsa.seasonal_decompose(ts, model='additive', period=365*96)
# decomposition.plot()


### rescaling
data_rescaled, k_factors = pre.rescaling(data, period_rescaling=96*365) #1 year
plot(data_rescaled, time, 2015, 2016)
plot(data_rescaled, time, 2018, 2019) 

### median
med = pre.median(data_rescaled)
fig = plt.figure(figsize=(8.0, 4.0))
plt.plot(time, med) ; plt.show()

### remove common signal 
ratio = pre.remove_signal(data, ref=med) 
plot(ratio, time, 2015, 2016) 

### rescale the ratio
ratio, k_factors = pre.rescaling(ratio, period_rescaling=96*365) 
plot(ratio, time, 2015, 2016)

k_tot = np.mean(k_factors, axis=0) 

plot(ratio, time, 2015, 2016, same_ax=True)
plot(ratio, time, 2017, 2018, same_ax=True) 
plot(ratio, time, 2018, 2019, same_ax=True) 


### select an IC pool
pool = pre.pool_clustering(ratio) # 17
names_IC = [names[i] for i in range(n_series) if i in pool]
names_OC = [names[i] for i in range(n_series) if i not in pool]

plot(ratio, time, 2015, 2020, ind=[1,20,12,14,17], same_ax=True) #OC
plot(ratio, time, 2015, 2020, ind=[0,5,10,18,19], same_ax=True) #IC
plot(ratio, time, 2015, 2020, ind=[2,21,19,3,20], same_ax=True) #IC and OC

ratioIC = ratio[:, pool]

### standardise the data
#K_knee = pre.choice_K(ratio, ratioIC, start=50, stop=2000, step=50)#200
#K_knee = pre.choice_K(ratio, ratioIC, start=500, stop=1200, step=50)#850
K = 850
data_stn, dataIC_stn = pre.standardisation(ratio, ratioIC, K)

plot(data_stn, time, 2015, 2020, ind=list(np.arange(5)), same_ax=True)
plot(data_stn, time, 2015, 2020, ind=list(np.arange(5,10)), same_ax=True)
plot(data_stn, time, 2015, 2020, ind=list(np.arange(10,15)), same_ax=True)
plot(data_stn, time, 2015, 2020, ind=list(np.arange(15,20)), same_ax=True)


### autocorrelation
acf.acf_pacf_plot(data_stn, which_display=3, max_cov=50) 
acf.acf_pacf_plot(data_stn, which_display=21, max_cov=50)
acf.acf_pacf_plot(data_stn, which_display=20, max_cov=50)

#=========================================================================
### design of the chart 
#=========================================================================

### choice of the block length 
#bb_length = acf.block_length_choice(dataIC_stn, 1, 100, 5)#11 
#bb_length = acf.block_length_choice(dataIC_stn, 10, 30, 1) #14
bb_length = 14

delta_init = 2 #intial value for the target shift size
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)

### adjust the control limits
# control_limit, delta_min = chart.shift_size(data_stn, pool, dataIC=dataIC_stn,
#                         L_plus=10, delta=delta_init, ARL0_threshold=ARL0, 
#                         block_length=bb_length, qt=0.5, 
#                         missing_values ='omit') #1.06 - 9.94

delta_target = 1

# control_limit = chart.limit_CUSUM(dataIC_stn, delta=delta_target,
#                 block_length=bb_length, missing_values='omit') #11.875

control_limit = 11.426

#=========================================================================
### train support vector classifier and regressor 
#=========================================================================

### select the length of the input vector 
wdw_length = svm.input_vector_length(dataIC_stn, delta_target, control_limit,
                                block_length=bb_length) #35-36

wdw_length = 36
   
### train and validate the models
std_data = np.nanstd(data_stn)
scale = 2 #scale parameter (~variance) of the halfnormal distribution
n = 21000*3 #number of testing and training instances
n_search = 12000*3 #number of testing and training instances

### find an optimal value for C (regularization parameter)
# C_choices = svm.choice_of_C(dataIC_stn, control_limit, delta_target, wdw_length, scale,
#                     start = 5, stop = 15, step = 1,
#               delay=True, n=n_search, block_length=bb_length, confusion=False)

# C = C_choices[2] ##13

C = 10

### train the classifier and regressor with selected C and kernel
reg, clf = svm.training_svm(dataIC_stn, control_limit, delta_target,
                wdw_length, scale, delay=True, n=n, C=C, block_length=bb_length)

### save models 
# filename = 'svr_elia_min.sav'
# pickle.dump(reg, open(filename, 'wb'))
# filename = 'svc_elia_min.sav'
# pickle.dump(clf, open(filename, 'wb'))

### or load the models previously trained
reg = pickle.load(open('../svm_models/svr_elia_min.sav', 'rb'))
clf = pickle.load(open('../svm_models/svc_elia_min.sav', 'rb'))

#=========================================================================
### run the control chart and plot results (with svm predictions)
#=========================================================================

region = [i for i in range(len(names)) if names[i] == 'Regie de Wavre'][0]

 
for i in range(n_series): 
    data_indv = data_stn[:,i]
    
    [form_plus, form_minus, size_plus, size_minus,
    C_plus, C_minus] = appl.alerts_info(data_indv, control_limit, 
            delta_target, wdw_length, clf, reg)
                                   
    fig = appl.plot_3panels(data_indv, control_limit, time, 
                     form_plus, form_minus, size_plus, size_minus, 
                     C_plus, C_minus, names[i], time_start=2015,
                     time_stop=2016)
    
    fig = appl.plot_4panels(data_indv, data[:,i], control_limit, time, 
                  form_plus, form_minus, size_plus, size_minus, 
                  C_plus, C_minus, names[i], time_start=2015,
                  time_stop=2021)
        
    fig = appl.plot_1panel(data_indv, time, 
              form_plus, form_minus, size_plus, size_minus, names[i],
              time_start=2016, time_stop=2017)

    ##fig.savefig('../figures/%s_min.pdf' %names[i]) #save figures
    
    
#March 20, 2015 
region = [i for i in range(len(names)) if names[i] == 'Regie de Wavre'][0]


for i in range(region, region+1): 
    data_indv = data_stn[:,i]
    
    [form_plus, form_minus, size_plus, size_minus,
    C_plus, C_minus] = appl.alerts_info(data_indv, control_limit, 
            delta_target, wdw_length, clf, reg)
                                   
    fig = appl.plot_3panels(data_indv, control_limit, time, 
                     form_plus, form_minus, size_plus, size_minus, 
                     C_plus, C_minus, names[i], time_start=time[7488],
                     time_stop=time[7576], years=False, hours=True)
    
     ##fig.savefig('../figures/eclipse.pdf') #save figures
     
     
#August 30, 2019 IMEA
region = [i for i in range(len(names)) if names[i] == 'IMEA'][0]

for i in range(region, region+1): 
    data_indv = data_stn[:,i]
    
    [form_plus, form_minus, size_plus, size_minus,
    C_plus, C_minus] = appl.alerts_info(data_indv, control_limit, 
            delta_target, wdw_length, clf, reg)
                                  
    
    fig = appl.plot_3panels(data_indv, control_limit, time, 
                 form_plus, form_minus, size_plus, size_minus, 
                 C_plus, C_minus, names[i], time_start=time[163392-96],
                 time_stop=time[163487+96+96], years=False, hours=True)
    
     ##fig.savefig('../figures/IMEA_2019_v2.pdf') #save figures
     
     
#September 14, 2019 IMEA
region = [i for i in range(len(names)) if names[i] == 'IMEA'][0]

for i in range(region, region+1): 
    data_indv = data_stn[:,i]
    
    [form_plus, form_minus, size_plus, size_minus,
    C_plus, C_minus] = appl.alerts_info(data_indv, control_limit, 
            delta_target, wdw_length, clf, reg)
                                
    
    fig = appl.plot_3panels(data_indv, control_limit, time, 
                 form_plus, form_minus, size_plus, size_minus, 
                 C_plus, C_minus, names[i], time_start=time[164832],
                 time_stop=time[164927], years=False, hours=True)