
# cusvm

This folder contains the main functions of the package. 

## Folder contents


* **preprocessing** may be used to preprocess the data. It includes functions to remove the individual levels of the series or smooth the series by a moving 
average filter. It also contains functions to select a pool of stable series from the panel, estimate the longitudinal IC patterns of the data and standardise the data by these patterns.

* **bb_methods** contains functions related to the block boostrap (BB) procedures. Four block bootstrap methods are implemented: moving block bootstrap (MBB), circular block bootstrap (CBB), non-overlapping block bootstrap (NBB) and matched block bootstrap (MABB). 

* **autocorrelations** is composed of several functions that compute the autocorrelation (acf) and partial autocorrelation functions (pacf) of time-series with missing observations. 
It also contains functions to graphically display the acf and pacf until a specified lag and to automatically select an appropriate value for the block 
length of the BB procedures.

* **cusum_design_bb** contains a set of functions to design the CUSUM chart using a BB procedure. In particular, functions for computing the in-control average run length (ARL0) and out-of-control average run length (ARL1) of the chart are implemented as well as an algorithm to adjust the control limits of the chart.

* **svm_training** includes several functions related to the support vector machine (SVM) procedures. Those automatically estimate an appropriate value for the length of the input vector or train and validate the SVMs.

* **alerts** contains a function that actually applies the monitoring based on the CUSUM chart to the series. It returns the predicted sizes and shapes of the deviations after each alert. Other functions are also dedicated to graphically display the main features of the monitoring procedure. 




