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

## Notebooks interpretation

 1. [EDA](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-EDA.ipynb)
 2. [PREPROCESSING](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-PREPROCESSING.ipynb)
 3. [MODEL_SELECTION](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-MODEL_SELECTION.ipynb) - performs cross-validation of models to select the best one;
 4. [TUNING](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-TUNING.ipynb) - tunes the selected models hyperparameters, to improve score;
 5. [GENERAL](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-GENERAL.ipynb) - joins all the steps and performs predictions.
 
<!-- end of the list -->
 
 ## Solution Framework
 
 In order to solve the problem I have applied the following Data Science mindset:

1. Explore the data using [EDA](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-EDA.ipynb) - gain insight on the main aspects of the data such as distributions, trends, predictors, etc.
2. Clean data in [PREPROCESSING](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-PREPROCESSING.ipynb) - apply the gained insight to preprocess the data and getting it ready for model consumption.
3. Perform cross validation of [MODELS](https://gitlab.com/Jpsr2/drivendata-flu-shot-learning/-/blob/master/flu_shot_learning-MODEL_SELECTION.ipynb) - select the models I want to use; these models will be a basis to test different preprocessing assumptions and will eventually be part of the final model;
4. Tune some of the models using [OPTUNA](https://optuna.org/);
5. Get everything together and make predictions;
6. Iterate through every step applying different preprocessing assumptions, model building techniques and trying to optimize the model to the AUROC metric.

<!-- end of the list -->