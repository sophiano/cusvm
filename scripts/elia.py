   # -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:21:43 2020

@author: sopmathieu

This script applies the monitoring procedure on data obtained each 15 minutes. 

"""

import pickle
import numpy as np 
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.rcParams['font.size'] = 14
import sys
sys.path.insert(1, '../')

from cusvm import preprocessing as pre
from cusvm import errors_extraction as err
from cusvm import cusum_design_bb as chart
from cusvm import alerts as plot
from cusvm import svr_svc_training as svm
#from cusvm import block_length as bbl

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

# ### load data
# with open('data/data_regions', 'rb') as file:
#     my_depickler = pickle.Unpickler(file)
#     mc = my_depickler.load() 
#     data = my_depickler.load() 
#     date = my_depickler.load() 
# (n_obs, n_series) = data.shape    
    
# date_obj = []
# time = []
# for i in range(n_obs):
#     date_obj.append(datetime.strptime(date[i], '%Y-%m-%d-%H:%M:%S'))
#     time.append(toYearFraction(date_obj[i]))

# # export data (time in fraction of years)
# with open('data/time', 'wb') as file: 
#       my_pickler = pickle.Pickler(file)
#       my_pickler.dump(time) 

#=============================================================
#=============================================================   
    
### load data
with open('../data/data_regions', 'rb') as file:
    my_depickler = pickle.Unpickler(file)
    mc = my_depickler.load() 
    data = my_depickler.load() #regions 4 -> 14
    date = my_depickler.load() 
(n_obs, n_series) = data.shape    
       

with open('../data/regions', 'rb') as file:
    my_depickler = pickle.Unpickler(file)
    names = my_depickler.load() 
 
with open('../data/time', 'rb') as file:
    my_depickler = pickle.Unpickler(file)
    time = my_depickler.load() #data every 15 min.


region_names = []
region_id = []
for i in range(len(names)):
    if int(names[i]['SourceId'])>3 and int(names[i]['SourceId'])<15:
        region_names.append(names[i]['Name'])
        region_id.append(int(names[i]['SourceId']))
num = np.argsort(np.array(region_id))
region_names = [region_names[i] for i in num] #in order 4-> 14

time = np.array(time) #96 obs per day

#=============================================================
#=============================================================

#remove zero (nights)
ind = np.where(data[:,0]!=0)[0]
data_wht_zero = np.zeros((n_obs, n_series))
data_wht_zero[:] = np.nan
data_wht_zero[ind,:] = data[ind, :]
data = np.round(data_wht_zero)

data_rescaled = err.rescaling(data, period_rescaling=1000) #10 jours
data = err.ratio(data_rescaled)

#data = err.error(data, period_rescaling=1000)
#Mt = err.median(data_rescaled)

max_pos = np.zeros((n_series)); max_val = np.zeros((n_series))
for i in range(n_series):
    ind = np.where(np.isnan(data[:,i]) == False)[0]
    sub = data[:,i]
    max_val[i] = np.max(sub[~np.isnan(sub)])
    max_pos[i] = ind[np.argmax(sub[~np.isnan(sub)])]

#plot all data
plt.hist(data[~np.isnan(data)], range=[-4,4], bins='auto', density=True, facecolor='b')  
plt.title("All Data")
print(np.nanmean(data))
print(np.nanstd(data))
plt.axis([-4, 4, 0, 1.5])
plt.grid(True)
plt.show()

#=============================================================
#=============================================================
    

start_month = datetime.strptime('2016-01-01-00:00:00', '%Y-%m-%d-%H:%M:%S')
start = np.where(time >= toYearFraction(start_month))[0][0] 
stop_month = start_month + relativedelta(months=1)
stop = np.where(time >= toYearFraction(stop_month))[0][0] 
#np.where(time >= 2019.1)[0][0]
if stop-start < 4000:
    ind_ticks = np.arange(start, stop, 96) #one day
    x_ticks = np.round(time[ind_ticks],2)
else :
    ind_ticks = np.arange(start, stop, 96*30*2) #two month
    x_ticks = np.round(time[ind_ticks],2)
    
    
count = 1
fig = plt.figure()
for i in [0, 3, 4, 5, 10]:
    f = fig.add_subplot(5, 1, count)
    plt.ylabel(region_names[i])
    #plt.plot(data[start:stop, i])
    plt.plot(time[start:stop], data[start:stop, i])
    if count < 5:
        f.axes.get_xaxis().set_ticklabels([]) 
    #f.set_ylim([0,5]); f.set_xlim([time[start], time[stop]])
    plt.xticks(x_ticks)
    count += 1
plt.show()


#=============================================================
#=============================================================
    

### Apply preprocessing
data_elia = pre.PreProcessing(data)  
data_elia.level_removal(level=False) #wdw=6000)  #level since repartition is not uniform (1=>0)
data_elia.selection_pools(method='kmeans', ref=None) #select the IC pool
pool = np.array(data_elia.pool) #pool (all except west flanders, liege, luwxembourg)
pool_ind = [region_names[i] for i in range(n_series) if i in pool] #pool (index)
data_elia.outliers_removal(k=3) #remove outliers

## Choose an appropriate value for K
dataIC = data_elia.dataIC  #IC data without deviations
data = data_elia.data #data (IC and IC) with deviations
K = pre.choice_K(data, dataIC, plot=True, start=50, stop=2000, step=50)#min a 500
#selected 150
#K = pre.choice_K(data, dataIC, plot=True, start=20, stop=40, step=1)

## Standardisation
data_elia.standardisation(K=300) #standardisation of the data
dataIC = data_elia.dataIC  #IC data without deviations
data = data_elia.data #data (IC and IC) with deviations
     
#plot the (IC) data
plt.hist(dataIC[~np.isnan(dataIC)], range=[-4,4], bins='auto', density=True, facecolor='b')  
plt.title("Data IC")
plt.text(2, 1, 'mean:' '%4f' %np.nanmean(dataIC))
plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(dataIC))
plt.axis([-4, 4, 0, 1.5])
plt.grid(True)
plt.show()

#plot all data
plt.hist(data[~np.isnan(data)], range=[-4,4], bins='auto', density=True, facecolor='b')  
plt.title("All Data (IC and OC)")
plt.text(2, 1, 'mean:' '%4f' %np.nanmean(data))
plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(data))
plt.axis([-4, 4, 0, 1.5])
plt.grid(True)
plt.show()

n_obs_IC = len(dataIC[~np.isnan(dataIC)])*100 / len(data[~np.isnan(data)])#73%

#=========================================================================
#=========================================================================
### design of the chart 

#block length 
#large range
#bb_length = bbl.block_length_choice(dataIC[:,0:3], wdw_min=1, wdw_max=100, wdw_step=10) #21
#smaller range
#bb_length = bbl.block_length_choice(dataIC, wdw_min=15, wdw_max=25, wdw_step=1) #19
bb_length = 20

delta_min = 1 #intial value for the target shift size
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)

### Adjust the control limits
control_limit, delta_min = chart.reccurentDesign_CUSUM(data, pool, dataIC=dataIC,
                delta=delta_min, ARL0_threshold=ARL0, block_length=bb_length, 
                qt=0.5, missing_values ='omit') #1.99-6.74
#

delta_min = 2
control_limit = chart.search_CUSUM_MV(dataIC, delta=delta_min, ARL0_threshold=200,
                block_length=bb_length, missing_values='omit') #7.14-7.1
control_limit = 7.14 #fixed

# warning_limit = chart.search_CUSUM_MV(dataIC, delta=delta_min, ARL0_threshold=200,
#                block_length=bb_length, missing_values='reset') #

### Compute the performance of the chart
# ARL1 = chart.ARL1_CUSUM_MV(dataIC, control_limit, delta=delta_min, 
#                            missing_values='reset', block_length=bb_length) #


#=========================================================================
#=========================================================================
### train classifier and regressor 

### select the length of the input vector 
wdw_length = svm.input_vector_length(dataIC, delta_min, control_limit,
                                block_length=bb_length) #24-25
wdw_length=24
   
### train and validate the models
scale = 2.5 #scale parameter (~variance) of the halfnormal distribution
n = 21000*3 #number of testing and training instances
n_search = 12000*3 #number of testing and training instances

#find an optimal value for C (regularization parameter)
C_choices = svm.choice_of_C(dataIC, control_limit, delta_min, wdw_length, scale,
                    start = 5, stop = 15, step = 1,
              delay=True, n=n_search, block_length=bb_length, confusion=False)
C = C_choices[2] ##13


#train the classifier and regressor with selected C and kernel
reg, clf = svm.training_svr_svm(dataIC, control_limit, delta_min,
                wdw_length, scale, delay=True, n=n, C=C, block_length=bb_length)

# ## save models 
# filename = 'svr_elia.sav'
# pickle.dump(reg, open(filename, 'wb'))
# filename = 'svc_elia.sav'
# pickle.dump(clf, open(filename, 'wb'))

## or load the models previously trained
reg = pickle.load(open('models/svr_elia.sav', 'rb'))
clf = pickle.load(open('models/svc_elia.sav', 'rb'))


#=========================================================================
#=========================================================================
### run the control chart and plot results (with predictions)

region = 10

#for i in range(region, region+1): 
for i in range(11): 
    data_indv = data[:,i]
    level_indv = data_rescaled[:,i]
    
    [form_plus, form_minus, size_plus, size_minus,
    C_plus, C_minus] = plot.alerts_info(data_indv, control_limit, 
            delta_min, wdw_length, clf, reg)
    
    size_minus = size_minus                                
    fig = plot.plot_monitoring(data_indv, level_indv, control_limit, time, 
                     form_plus, form_minus, size_plus, size_minus, 
                     C_plus, C_minus, region_names[i], time_start=2020)
        
    fig.savefig('figures/%s_2020.pdf' %region_names[i]) #save figures

