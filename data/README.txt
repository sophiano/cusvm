===================
data
===================

**data_regions** contains the data from elia. 

'mc' == The sum of monitored capacities for all selected solar regions (MonitoredCapacity)
'data' == The sum of real time measures for all selected solar regions (RealTime)
'date' == represente the datetime of the measure (from January 1, 2016 to January 1, 2020)

====================================

**regions** contains the name of the regions that are analyzed and their code names.

====================================

**time** contains the time of the measures expressed in fraction of years.

======================================

The files are provided into a serialized format, which may be opened with the python module 'pickle' with: 

with open('data/data_regions', 'rb') as file:
    my_depickler = pickle.Unpickler(file)
    mc = my_depickler.load() 
    data = my_depickler.load() 
    date = my_depickler.load() 

(with appropriate paths).


