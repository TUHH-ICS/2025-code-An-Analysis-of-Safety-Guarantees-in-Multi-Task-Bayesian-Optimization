# An Analysis of Safety Guarantees in Multi-Task Bayesian Optimization

## General

This repository contains supplementary material and the code to reproduce the tables and figures presented in 

> J. O. Lübsen, A. Eichler, "An Anlysis of Safety Guarantees in Multi-Task Bayesian Optimization"


To run the proposed SaMSBO alogrithm the use may use the 'run_SaMSBO.py' file. To run the safe UCB algorithm for comparison, the user may use the 'run_SafeUCB.py' file.
In addition, there is a jupyter notebook 'visualized_example.ipynb' which shows the optimization of a one-dimensional example. This may be run to visually follow the optimization procedure.

The data used for plots to generate the figures in the manuscript are in the data folder. The use may use the 'generate_plots.ipynb' notebook to recreate the figures in the 'plot_scripts' folder.

## Prerequisites

To run the code install python3.12.8 and the dependencies specified in `requirements.txt`.

> pip install -r requirements.txt

The code in this repository was tested in the following environment:

* *Ubuntu 24.04.2 LTS
* *Python 3.12.8





