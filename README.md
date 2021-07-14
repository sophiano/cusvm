
# Module cusvm 

cusvm is a module written in Python to monitor panel of time-series data.

The package contains different functions to apply a robust non-parametric quality control based on the block bootstrap and the CUSUM chart to a panel of series. The method also provides predictions of the shapes and the sizes of the deviations encountered, using support vector machine (SVM) procedures. <br>
The name _cusvm_ comes from the contraction of the two main ingredients of the method: the CUSUM chart and the SVM methods.

This package was written by Sophie Mathieu (UCLouvain/Royal Observatory of Belgium). 

## Installation 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install cusvm:

````
pip install git+https://github.com/sophiano/cusvm
````
Tips : We strongly adsive the user to install the package in a new Python environment (with a version of Python superior or equal to 3.6) to avoid any conflict with existing packages. For more information about the installation, we refer to the setup. 

## Folder contents

* **cusvm** <br>
The main functions of the package are contained in the folder cusvm.

* **data** <br>
The data that are used as one example of application of the method are located
in the folder data.

* **docs** <br>
This folder contains the doumentation of the package in the form of Jupyter notebooks. 

* **scripts** <br>
This folder contains the scripts of the package.

* **svm_models** <br>
The trained support vector machine classifiers and regressors are saved into the folder svm_models. They are conserved for reproducibility purposes. 


## References

* Mathieu, S., Lefèvre, L, von Sachs, R., Delouille, V., & Ritter, C. (2020).
_Nonparametric robust monitoring of time series panel data_.
Available on [arXiv](https://arxiv.org/abs/2010.11826).

* Mathieu, S., Lefèvre, L, von Sachs, R., Delouille, V., Ritter, C. & Clette, F. (2021).
_Nonparametric monitoring of sunspot number observations: a case study_.
Available on [arXiv](https://arxiv.org/abs/2106.13535).

## License
[MIT](https://choosealicense.com/licenses/mit/)



