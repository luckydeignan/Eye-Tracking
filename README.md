# Eye-Tracking
Framework for conducting eye-tracking experiments and data analysis using Tobii Pro Spark

## Introduction
This software provides a framework conducting eye-tracking experiments with the Tobii Pro Spark and analyzing the data. This code is specifically designed to analyze humans' eye movements when answering questions related to the [DocVQA dataset](https://rrc.cvc.uab.es/?ch=17&com=downloads), specifically for Task 1: "Single Page Document Visual Question Answering".

There are three files of code associated with this software, two written in Python and another serving as a JSON -- *experiment_data.json* -- storing the data for the eye tracking experiment. **main_GUI.py** serves as the user-interface and actual experiment itself, recording eye-tracking data along with displaying each of the image documents along with their corresponding question. **main_data_analysis.py** can then be used to fetch data from the JSON to analyze it.

The functionality of the data analysis -- via analyze_data function -- includes creating a heatmap to allow for visual representation of the subject's gaze upon the image throughout their time answering the question, and displaying time-space clusters on a scatterplot, each representing a fixation point upon the image. Finally, the function also returns the nearest OCRs for each cluster within a specified distance (a hyperparameter determined by the experimenter).

## Installation
Create a virtual environment in Python 3.8 in order to collect data from the Tobii Pro Spark.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary packages for your virtual environment.

```bash
pip install tobii_research
pip install matplotlib
pip install hdbscan
pip install sklearn
pip install scipy
pip install pillow
pip install tkinter
pip install numpy
```

Download the [Tobii Pro Eye Tracker Manager](https://connect.tobii.com/s/etm-downloads?language=en_US). Follow on-screen instructions once installed to configure your Tobii Pro Spark to your device.

(Note: there is currently an unresolved issue with successfully installing this software to Linux devices, ongoing efforts are currently being done to resolve this problem).

## Usage
Three are two files to run:

**main_GUI.py**: runs the eye tracking experiment, and then saves the data to the JSON within the directory.

**main_data_analysis.py**: using the analyze_data() function, creates visual heatmap of a specific image/question pair, and returns all the OCRs within a specified distance of each visual fixation point.

In order to maximize value from this software and analysis, it is important to note the several hyperparameters within the code and their importance.

### Hyperparameters
- **main_GUI.py**
  - *folder_path*: the file path on your system that contains the images that you'd like to display for your experiment
  - *ocr_data*: in the context of SQVA, loads the OCR imbd data based on the path given from your device
  - *reduced_prop*: variable under *fit()* function. Determines the maximum proportion of width/height of screen that image       will take up when being displayed on screen.
- **main_data_analysis.py**
  - *rad*: variable under *nearest_ocr* function. Determines radius around fixation point in which OCR's are fetched
  - *min_cluster_size*: variable under *ocr_cluster* function. Determines minimum number of points to create a cluster from       eye-tracking data. More info in file comments.
