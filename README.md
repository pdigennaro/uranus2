# URANUS

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0) ![version](https://img.shields.io/badge/version-1.0-brightgreen)


This repository contains the source code for the URANUS framework.
URANUS is an ML framework used to identify, classify and track Unmanned Aerial Vehicles (UAV).\
For the identification and classification tasks of a UAV, a MLP model is trained and tested, with a final accuracy of 90%. For the tracking task, a single regressor Random Forest model, is deployed to provide the exact position of the UAV, 
with MSE of 0.29, MAE of 0.04, R^2 of 0.93.

<p align="center">
  <img src="results/schemas/schema.PNG" width="80%" height="80%">
</p>

The libraries used by URANUS are the following:

| Plugin                | README                            |
|-----------------------|-----------------------------------|
| PyTorch               | [https://pytorch.org/]            |
| sklearn.preprocessing | [https://scikit-learn.org/stable] |
| NumPy                 | [https://numpy.org/]              |
| Pandas                | [https://pandas.pydata.org/]      |
| Matplotlib            | [https://matplotlib.org/]         |

Screenshot
-----

This is a screenshot of the final application:
<p align="center">
  <img src="results/screens/demo-small.PNG" width="50%" height="50%">
</p>

Instructions
------
The proposed source files have been developed and tested with Ubuntu 22.04 LTS and Python 3.11 and we encourage to use a similar environment to run the scripts.

Use the following commands to install the scripts' dependencies:

```sh
$ sudo apt-get install python3-pip python3-tk graphviz
$ pip3 install -r requirements.txt
```

License
----

URANUS is released under the GPLv3 <a href="LICENSE">license</a>.
