# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:54:33 2021

@author: sopmathieu
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


### load data 
df = pd.read_csv (r'../data/PVq.csv')
data = np.array(df)[4:,1:] 
data = data.astype('float')
names = list(df.columns)[1:]
(n_obs, n_series) = data.shape   


### plot all data
# plt.hist(data[~np.isnan(data)], bins=120, density=True, facecolor='b')  
# plt.title("Elia data (15 min)")
# plt.grid(True)
# plt.axis([-1, 120, 0, 0.1])
# plt.ylabel('Density')
# plt.xlabel('load factor')
# plt.savefig('hist_min.pdf') #save figures
# plt.show()

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
        ind_ticks = np.arange(start, stop, 180*96) #six months
        x_ticks = np.round(time[ind_ticks],1)
        
    max_val = max(data[~np.isnan(data)])
    min_val = min(data[~np.isnan(data)])
    y_max = 0.75*max_val
    y_min = 0.75*min_val
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

def plot_single(data, time, start_time=2015, stop_time=2021, ind=3, same_ax=False):
    
    start = np.where(time >= start_time)[0][0] 
    stop = np.where(time >= stop_time)[0][0] 
    if stop-start < 1000:
        ind_ticks = np.arange(start, stop, 60*96) #two months
        x_ticks = np.round(time[ind_ticks],2)
    else :
        ind_ticks = np.arange(start, stop, 365*96) #one year
        x_ticks = np.round(time[ind_ticks])
        
        
    fig = plt.figure(figsize=(8.0, 4.5))
    plt.title(names[ind])
    plt.xlabel('year')
    plt.plot(time[start:stop], data[start:stop, ind])
    plt.xticks(x_ticks)
    
    return fig

plot(data, time, 2015, 2016) 
plot(data, time, 2019, 2020) 

plot(data, time, 2017.2, 2017.5, [1,2,6,7,16])
plot(data, time, 2017, 2018) 


#=====================================================================
### Histograms and graphs for the preprocessing
#======================================================================

### remove zero (nights)
(n_obs, n_series) = data.shape  
ind = np.where(data[:,0]!=0)[0]
data_wht_zero = np.zeros((n_obs, n_series))
data_wht_zero[:] = np.nan
data_wht_zero[ind,:] = data[ind, :]
data = data_wht_zero

### plot all data
# plt.hist(data[~np.isnan(data)], bins='auto', density=True, facecolor='b')  
# plt.title("Elia data (15 min))")
# plt.grid(True)
# plt.axis([-1, 140, 0, 0.11])
# plt.show()

region = [i for i in range(len(names)) if names[i] == 'IMEA'][0]

fig = plot_single(data, time, ind=region)
plt.ylabel('$P(i,t)$ ')
#plt.savefig('flow_1_el_min.pdf') #save figure
plt.show()

plt.hist(data[:,region], bins='auto', density=True, facecolor='b')  
plt.text(40, 0.04, 'mean: ' '%.3f' %np.nanmean(data[:,region]))
plt.text(40, 0.03, 'std: ' '%.3f' %np.nanstd(data[:,region]))
plt.xlabel('$P(i,t)$')
plt.title(names[region])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_1_el°min.pdf') #save figure
plt.show()

#=====================================================================

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

fig = plot_single(ratio, time, ind=region)
plt.ylabel('$P(i,t)/\hat c(t)$ ')
#plt.savefig('flow_2_el_min.pdf') 
plt.show()

plt.hist(ratio[:,region], bins='auto', density=True, facecolor='b')  
plt.text(1.5, 3, 'mean: ' '%.3f' %np.nanmean(ratio[:,region]))
plt.text(1.5, 2, 'std: ' '%.3f' %np.nanstd(ratio[:,region]))
#plt.axis([0,150, 0, 0.08])
#plt.yticks(np.arange(0, 0.02, 0.005))
plt.xlabel('$P(i,t)/\hat c(t)$')
plt.title(names[region])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_2_el_min.pdf') 
plt.show()

### rescale the ratio
ratio, k_factors = pre.rescaling(ratio, period_rescaling=96*365) 
k_tot = np.mean(k_factors, axis=0) 

fig = plot_single(ratio, time, ind=region)
plt.ylabel('$\hat \mu_{\eta}(i,t)$ ')
#plt.savefig('flow_3_el_min.pdf')
plt.show()

plt.hist(ratio[:,region], bins='auto', density=True, facecolor='b')  
plt.text(1.5, 2.5, 'mean: ' '%.3f' %np.nanmean(ratio[:,region]))
plt.text(1.5, 2, 'std: ' '%.3f' %np.nanstd(ratio[:,region]))
#plt.axis([0,150, 0, 0.08])
#plt.yticks(np.arange(0, 0.02, 0.005))
plt.xlabel('$\hat \mu_{\eta}(i,t)$')
plt.title(names[region])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_3_el_min.pdf') 
plt.show()

### select an IC pool
pool = pre.pool_clustering(ratio) # 17
names_IC = [names[i] for i in range(n_series) if i in pool]
names_OC = [names[i] for i in range(n_series) if i not in pool]

plot(ratio, time, 2015, 2020, ind=[1,20,12,14,17], same_ax=True) #OC
plot(ratio, time, 2015, 2020, ind=[0,5,10,18,19], same_ax=True) #IC
plot(ratio, time, 2015, 2020, ind=[2,21,19,3,20], same_ax=True) #IC and OC

ratioIC = ratio[:, pool]

### standardise the data
#K_knee = pre.choice_K(ratio, ratioIC, start=50, stop=2000, step=50)
K = 850 
data_stn, dataIC_stn = pre.standardisation(ratio, ratioIC, K)

fig = plot_single(data_stn, time, ind=region)
plt.ylabel('$\hat \epsilon_{\eta}(i,t)$ ')
#plt.savefig('flow_4_el_min.pdf') 
plt.show()

plt.hist(data_stn[:,region], bins='auto', density=True, facecolor='b')  
plt.text(7.5, 0.3, 'mean: ' '%.3f' %np.nanmean(data_stn[:,region]))
plt.text(7.5, 0.2, 'std: ' '%.3f' %np.nanstd(data_stn[:,region]))
#plt.axis([0,150, 0, 0.08])
#plt.yticks(np.arange(0, 0.02, 0.005))
plt.xlabel('$\hat \epsilon_{\eta}(i,t)$')
plt.title(names[region])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_4_el_min.pdf') 
plt.show()

### plot the (IC) data
# plt.hist(dataIC_stn[~np.isnan(dataIC_stn)], range=[-4,4], bins='auto', density=True, facecolor='b')  
# plt.title("Data IC")
# plt.text(2, 1, 'mean:' '%4f' %np.nanmean(dataIC_stn))
# plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(dataIC_stn))
# plt.axis([-4, 4, 0, 1.5])
# plt.grid(True)
# plt.show()

### plot all data
# plt.hist(data_stn[~np.isnan(data_stn)], range=[-4,4], bins='auto', density=True, facecolor='b')  
# plt.title("All Data (IC and OC)")
# plt.text(2, 1, 'mean:' '%4f' %np.nanmean(data_stn))
# plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(data_stn))
# plt.axis([-4, 4, 0, 1.5])
# plt.grid(True)
# plt.show()


### autocorrelation
acf.acf_pacf_plot(data_stn, which_display=3, max_cov=50) 
acf.acf_pacf_plot(data_stn, which_display=21, max_cov=50)
acf.acf_pacf_plot(data_stn, which_display=20, max_cov=50)

