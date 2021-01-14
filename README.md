## BPANN_RelaySelection
Source code of the research "Combined Relay Selection Enabled by Supervised Machine Learning"

# Environment
This program was tested on CentOS 7.x and Ubuntu18/20.
Please make sure Tensorflow 2.1.0, Cuda 10.1 and Cudnn 7.6.4 were installed properly.

Some tips that might help:
1. To install Tensorflow 2.1.0 with pip:
> python -m pip install tensorflow==2.1.0
(Tensorflow 2.1.0 works with python 3.7 rather than 3.8, please make sure you have the correct python version :P )
2. Remember to add corresponding PATH according to instructions given by Cuda installer

# How to run it
To launch a new run:
python nn_relay.py -n

To continue a previous trainning:
python nn_relay.py -l [model name]
* the model name can be found under the "record" folder, named by the initial run datetime
