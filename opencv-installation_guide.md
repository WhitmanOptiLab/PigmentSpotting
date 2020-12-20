# How to install opencv 4 python
Python 3.7 or greater is required (https://www.python.org/)
## Installation using anaconda (recommended):
1. Install anaconda individual edition for your OS (https://docs.anaconda.com/anaconda/install/)
2. In terminal/command prompt, use the folowing commands:
    `conda create -- [your environment name] python=3.8` 
    `conda activate [your environment name]` 
    `conda install -c conda-forge opencv`

## Windows:
Instalation using pip:
1. In command prompt, use the following commands:
    `pip install --upgrade pip`
    `pip install numpy`
    `pip install opencv-python`

## Mac OSX: 
Installation using pip:
1. In terminal, use the following commands: 
    `pip install --upgrade pip`
    `pip install numpy` 
    `pip install opencv-python`

## Linux(Ubunutu)
1. In terminal, use the following commands:
    `wget wget https://bootstrap.pypa.io/get-pip.py`
    `sudo python3 get-pip.py`
    `pip install --upgrade pip`
    `pip install numpy`
    `pip install opencv-python`

## Verifying Installation
1. Enter the python prompt from terminal/command prompt with command `python`
2. import cv2
3. print(cv2.\_\_version\_\_)

If there are no errors you are all set to use opencv with python.
