# Data

This folder contains the datafiles. <br>
They are provided into csv files or serialized format (pickling).

## Folder contents

**PVdaily** contains the load factors (real time measures divided by the related monitored capacity) of the photovoltaic energy production in Belgium. 
They are provided by different distribution system operators (DSO), which are responsible for managing a small region (i.e. typically few municipalities) of the whole country. <br>
The data cover the years 2015-2020 and correspond to the sum of the load factors collected over a day. <br>

They are provided by [Elia](https://www.elia.be/fr).

**time_daily** contains the time of the observations, expressed in fraction of years. 
This file can be opened with the module 
[pickle](https://docs.python.org/3/library/pickle.html) (with appropriate path): 

````
with open('../data/time_daily', 'rb') as file:
    my_depickler = pickle.Unpickler(file)
    time = my_depickler.load() 
````



