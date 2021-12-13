## BPANN_RelaySelection<br/>
Source code of the research "Combined Relay Selection Enabled by Supervised Machine Learning"<br/>
<br/>
# Environment<br/>
This program was tested on CentOS 7.x and Ubuntu18/20.<br/>
Please make sure Tensorflow 2.1.0, Cuda 10.1 and Cudnn 7.6.4 were installed properly.<br/>
<br/>
Some tips that might help:<br/>
1. To install Tensorflow 2.1.0 with pip:<br/>
```
python -m pip install tensorflow==2.1.0
```
(Tensorflow 2.1.0 works with python 3.7 rather than 3.8, please make sure you have the correct python version :P )<br/>
2. Remember to add corresponding PATH according to instructions given by Cuda installer<br/>

Update(2021.12.13): Tested with Tensorflow 2.5.0. Should be compatible with later versions.

# How to run it
To launch a new run:<br/>
```
python nn_relay.py -n
```
To continue a previous trainning:<br/>
```
python nn_relay.py -l [model name]
```  
the model name can be found under the "record" folder, named by the initial run datetime

# Evaluates a trained model
There are various way to evaluate a trained model aquired in the previous process. We provide a quick and simple way to get some insight of the G matrices and predicted output vectors. Run nn_relay_verify.py as a stand-alone program:
```
python nn_relay_verify.py
```
The program would look for model "latest.h5" in the root directory and execute the evaluation function. While leaving the option export_to_files=True, the program generates two files in "verify" subfolder. "G.csv" and "results.csv" store the G matrices and predicted/brute-force searched results, respectively.
