# machine-learning-higgs 
Class project 1 | ML Higgs | EPFL<br/>


# Setup:
**Github setup:**<br/>
1. Copy this code to your clipboard: ```git clone https://github.com/StormFlaate/machine-learning-higgs.git```<br/>
2. Open your terminal on your computer, find a nice place and copy this code in to clone the repo<br/>
3. ```cd machine-learning-higgs```<br/>

**AIcrowd setup and collecting the last datafile:**<br/>
1. Create an account using your epfl.ch email and head over to the competition arena (https://www.aicrowd.com/challenges/epfl-machine-learning-higgs)
2. Download the dataset files from https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files
3. Unzip the files
4. Add the test.csv file into the ```/data``` (folder named data, can't add it here since github has a 100MB strict file size limit)
5. You are done for now!<br/>

# Structure:<br/>
**Folders:**<br/>
```/data/```-> contains 2/3 data files<br/>
```/latex-example-paper/```-> contains the latex example template and info<br/>

**Files:**<br/>
```/.gitignore```-> Includes all the files that git will ignore when pushing<br/>
```/README.md```-> Contains the information you are reading right now<br/>
```/project1_description.pdf```-> Includes the project 1 task description<br/>


# Physics Background:<br/>
The Higgs boson is an elementary particle in the Standard Model of physics which explains why other particles have mass. Its discovery at the Large Hadron Collider at CERN was announced in March 2013. In this project, you will apply machine learning techniques to actual CERN particle accelerator data to recreate the process of “discovering” the Higgs particle. For some background, physicists at CERN smash protons into one another at high speeds to generate even smaller particles as by-products of the collisions. Rarely, these collisions can produce a Higgs boson. Since the Higgs boson decays rapidly into other particles, scientists don’t observe it directly, but rather measure its“decay signature”, or the products that result from its decay process. Since many decay signatures look similar, it is our job to estimate the likelihood that a given event’s signature was the result of a Higgs boson (signal) or some other process/particle (background). In practice, this means that you will be given a vector of features representing the decay signature of a collision event, and asked to predict whether this event was signal (a Higgs boson) or background (something else). To do this, you will use the binary classification techniques we have discussed in the lectures.
If you’re interested in more background on this dataset, we point you to the longer description here: https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf.
