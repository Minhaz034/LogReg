## How to Regenerate ouputs / Run this project
Open terminal, navigate to the root directory and run:
```
./python main.py
```
## Outputs:
## Q1. 
Running the main.py file will generate `logistic_regression_costs.png` file that 
contains the outputs asked in Q1. The X axis contains the number of iterations
and the Y axis denotes the cost for various learning rates.This repository already
contains the generated cost graph in the root directory.

## Q2.
Q2 asks for the performance (accuracy) of :
 - Stochastic Gradient Descent
 - Mini batch Gradient Descent 
 - Batch Gradient Descent
in both the training and test dataset for 300,000 iterations and Lr = 0.1

Running the main.py file will automatically generate a table-like structure
in the terminal showing the accuracy scores for this 3 different gradient descent
methods for both of the datasets.

Note: As we are running 300,000 iterations the code completion is supposed to
take some minutes.


## Environment Dependencies:
All the allowed packages are listed in the `requiremets.txt` file.

Packages:
 - numpy==1.26.1
 - matplotlib==3.8.1

## Other files in this project:
 - `HonorCode.png`
 - `ProjectReport.pdf`