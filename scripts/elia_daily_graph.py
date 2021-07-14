# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 08:44:14 2021

@author: sopmathieu
"""

import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
import sys
sys.path.insert(1, '../')

from cusvm import preprocessing as pre
from cusvm import autocorrelations as acf

### load data 
df = pd.read_csv (r'../data/PVdaily.csv')
data = np.array(df)[:,1:]
data = data.astype('float')
data = data/96
names = list(df.columns)[1:]
(n_obs, n_series) = data.shape   


### plot raw data
# plt.hist(data[~np.isnan(data)], bins='auto', density=True, facecolor='b')  
# plt.title("Elia data (day)")
# plt.grid(True)
# plt.ylabel('Density')
# plt.xlabel('load factor')
# plt.savefig('hist_day.pdf') #save figures
# plt.show()


### load time
with open('../data/time_daily', 'rb') as file:
    my_depickler = pickle.Unpickler(file)
    time = my_depickler.load() #data every day
    
    
### remove year 2015 with unusual deviations
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
    return fig

def plot_single(data, time, start_time=2015, stop_time=2021, ind=3, same_ax=False):
    
    start = np.where(time >= start_time)[0][0] 
    stop = np.where(time >= stop_time)[0][0] 
    if stop-start < 1000:
        ind_ticks = np.arange(start, stop, 60) #two months
        x_ticks = np.round(time[ind_ticks],2)
    else :
        ind_ticks = np.arange(start, stop, 365) #one year
        x_ticks = np.round(time[ind_ticks])
        
        
    fig = plt.figure(figsize=(8.0, 4.5))
    plt.title(names[ind])
    plt.xlabel('year')
    plt.plot(time[start:stop], data[start:stop, ind])
    plt.xticks(x_ticks)
    
    return fig

################################################

plot(data, time, 2016, 2017) #RESA, IVEG, IMEA 2015
plot(data, time, 2019, 2020) #14 September, IMEA (pic)
plot(data, time, 2017, 2017.8) #8march-21May (par 1/4H, changements brusques, avec 0)

# plot_single(data, time, 2016, 2018, ind=5)
# plt.ylabel('$P(i,t)$')
# plt.savefig('mult.pdf') #save figures

region = [i for i in range(len(names)) if names[i] == 'IMEA'][0]

fig = plot_single(data, time, ind=region)
plt.ylabel('$P(i,t)$ ')
#plt.savefig('flow_1_el.pdf') #save figure
plt.show()

plt.hist(data[:,region], bins='auto', density=True, facecolor='b')  
plt.text(40, 0.04, 'mean: ' '%.3f' %np.nanmean(data[:,region]))
plt.text(40, 0.03, 'std: ' '%.3f' %np.nanstd(data[:,region]))
plt.xlabel('$P(i,t)$')
plt.title(names[region])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_1_el.pdf') #save figure
plt.show()


#=====================================================================
### Histograms and graphs for the preprocessing
#======================================================================

### rescaling
data_rescaled, k_factors = pre.rescaling(data, period_rescaling=365) 
plot(data_rescaled, time)

### median
med = pre.median(data_rescaled)
fig = plt.figure(figsize=(10.0, 6.0))
plt.plot(time, med) ; plt.show()

### remove common signal 
ratio = pre.remove_signal(data, ref=med) #mean = 1
plot(ratio, time)

fig = plot_single(ratio, time, ind=region)
plt.ylabel('$P(i,t)/\hat c(t)$ ')
#plt.savefig('flow_2_el.pdf') 
plt.show()

plt.hist(ratio[:,region], bins='auto', density=True, facecolor='b')  
plt.text(1.5, 4, 'mean: ' '%.3f' %np.nanmean(ratio[:,region]))
plt.text(1.5, 3, 'std: ' '%.3f' %np.nanstd(ratio[:,region]))
#plt.axis([0,150, 0, 0.08])
#plt.yticks(np.arange(0, 0.02, 0.005))
plt.xlabel('$P(i,t)/\hat c(t)$')
plt.title(names[region])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_2_el.pdf') 
plt.show()

### rescale the ratio
ratio, k_factors = pre.rescaling(ratio, period_rescaling=365)
plot(ratio, time)

fig = plot_single(ratio, time, ind=region)
plt.ylabel('$\hat \mu_{\eta}(i,t)$ ')
#plt.savefig('flow_3_el.pdf') 
plt.show()

plt.hist(ratio[:,region], bins='auto', density=True, facecolor='b')  
plt.text(1.5, 3, 'mean: ' '%.3f' %np.nanmean(ratio[:,region]))
plt.text(1.5, 2, 'std: ' '%.3f' %np.nanstd(ratio[:,region]))
#plt.axis([0,150, 0, 0.08])
#plt.yticks(np.arange(0, 0.02, 0.005))
plt.xlabel('$\hat \mu_{\eta}(i,t)$')
plt.title(names[region])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_3_el.pdf') 
plt.show()


### select an IC pool
pool = pre.pool_clustering(ratio) #10
names_IC = [names[i] for i in range(n_series) if i in pool]
names_OC = [names[i] for i in range(n_series) if i not in pool]

plot(ratio, time)
plot(ratio, time, ind=[2,7,12,14,17], same_ax=True) #OC
plot(ratio, time, ind=[9,11,18,19,21], same_ax=True) #IC
plot(ratio, time, ind=[1,21,2,8,15], same_ax=True) #IC and OC

ratioIC = ratio[:, pool]

### standardise the data
#K_knee = pre.choice_K(ratio, ratioIC, start=50, stop=2000, step=50)
K = 400
data_stn, dataIC_stn = pre.standardisation(ratio, ratioIC, K)
plot(data_stn, time)

fig = plot_single(data_stn, time, ind=region)
plt.ylabel('$\hat \epsilon_{\eta}(i,t)$ ')
#plt.savefig('flow_4_el.pdf') 
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
#plt.savefig('hist_4_el.pdf') 
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
acf.acf_pacf_plot(data_stn, which_display=2, max_cov=50)
acf.acf_pacf_plot(data_stn, which_display=21, max_cov=50) 

