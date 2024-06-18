# Plant Pigment Analyzer

## pipeline_script.py
Usage: `python pipeline_script.py <path to dataset directory> <path to result directory>`

The script expects that files in the dataset directory follow the pattern PLANTID_Vein_PETALID_DATE and PLANTID_Spot_PETALID_DATE, and that the result directory has alreacy been created.

## Setup and Dependencies
* Before you start, install: 
    * [Git](https://github.com/git-guides/install-git)
    * [Anaconda Python](https://www.anaconda.com/download/success)
* Use Git to clone this repository
* Use Anaconda to install the package scikit-learn
* Use Anaconda to create and activate a python3 environment for this project
* Within that environment, use pip to install opencv and rawpy packages
    * This project requires OpenCV 4.4+, and the developers have found the most consistent success with installing the pip package opencv-contrib-python

