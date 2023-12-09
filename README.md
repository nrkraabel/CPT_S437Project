# CPT_S437Project

Collab notebook: https://colab.research.google.com/drive/1m_Yr91WujioZTPY4afZjxPSCj70qk51q?usp=sharing#scrollTo=2PEZvLdmwA9-

Prensentation: https://emailwsu-my.sharepoint.com/:p:/r/personal/nicholas_kraabel_wsu_edu/Documents/Project_Presentation_437.pptx?d=w2a0044d02b684c6fb9566f06dd150488&csf=1&web=1&e=WvYLsl

Video Link: https://youtu.be/YdQCaCQVJ0k

#DataHandler.py
This is our data processing code used to load, save and extract file paths, coordinates

#GPS_Long_Lat_Compass.mat
Used to get Latitude and longitude of each image

#Graphics.ipynb
Juptyer notebook using folium package to generate maps of model predictions vs actual for the report

#Kamiak.slurm
Script to run model training Kamiak Wsu's super computer

#model.py
Contains all model definitions and dataloader class

#NoneDescriptCalculation.cpp
A simple calculation to determine amount of none useable image due to lack of varience within image

#RayTune.py
Ray tune package is used to optimize hyperparameters by running experiments adjust the hypermeters for several training sessions to determine the best training condtions

#Runner.ipynb
Similar to Kamiak.slurm it contains training scripts for the models however this
