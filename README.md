# Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines

The aim of this project is to predict the probability of a subject taking the H1N1 and Seasonal flu vaccines according to the provided data.
This project is built for the data science competition: https://www.drivendata.org/competitions/66/flu-shot-learning/.

In this readme, I will explain the steps I took to achieve my results in the competition:

 - [AUROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) = 0.8442
 - Top 14% of participants - as of writing this
   
## Repository Structure

Sub-folders: 
 - [*input_data*](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/tree/master/input_data) - raw data from competition;
 - [*interim_data*](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/tree/master/interim_data) - preprocessed data to be used in modelling;
 - [*output_data*](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/tree/master/output_data) - model predictions to submit;
 - [*models*](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/tree/master/models) - *pickled* models to be imported by the notebooks.
 
Main folder:
 - All the notebooks (*.ipynb*) and respective scripts (*.py*) for the project;
 - *requirements.txt* - project dependencies;
 - *report.pdf* - report on the solution thought process, implementation and results.

## Notebooks interpretation

 1. [EDA](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-EDA.ipynb) - provides data insights;
 2. [PREPROCESSING](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-PREPROCESSING.ipynb) - applies insights to data preprocessing;
 3. [MODEL_SELECTION](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-MODEL_SELECTION.ipynb) - performs cross-validation of models to select the best one;
 4. [TUNING](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-TUNING.ipynb) - tunes the selected models hyperparameters, to improve score;
 5. [GENERAL](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-GENERAL.ipynb) - joins all the steps and performs predictions.