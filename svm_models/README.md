
# svm_models

This folder contains the trained support vector machine classifiers (**svc**) and regressors (**svr**) for monitoring the daily photovoltaic energy production
in Belgium. 

The confusion matrices (normalized and not normalized) and the prediction performances of the models are also saved (in PNG images). 

Those models are saved for reproducibility purpose.

## Instructions

These models can be opened with the module 
[pickle](https://docs.python.org/3/library/pickle.html) (with appropriate path):  

````
reg = pickle.load(open('svm_models/svr_elia_daily.sav', 'rb'))
clf = pickle.load(open('svm_models/svc_elia_daily.sav', 'rb'))

````