import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setuptools.setup(
    name="cusvm", 
    version="0.0.2",
    author="Sophie Mathieu",
    author_email="sph.mathieu@gmail.com",
    description="Monitoring of panel of time-series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sophiano/cusvm",
    packages=['cusvm', 'data'], #setuptools.find_packages(),
    install_requires=['numpy>=1.19.5', 
                      'keras>=2.4.3',
                      'scipy>=1.6.0',
                      'matplotlib>=3.3.3',
                      'scikit-learn>=0.24.1',
                      'statsmodels>=0.12.1', 
                      'kneed>=0.7.0'
                      ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)