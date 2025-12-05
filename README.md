Machine Learning Assignments & Model Implementations

This repository contains a collection of machine learning assignments implemented in both Python and R.
The code focuses on fundamental models—neural networks, classical ML classifiers, ensemble methods, and Naive Bayes text classification—implemented clearly and transparently for learning and experimentation.

📁 Repository Contents
ML_model_implementations/
├── CNN_NNimplementation.py          # Neural network / CNN classifier (Python)
├── DTClassifier_implementation      # Decision Tree classifier (Python)
├── ensemble_classifier.R            # Ensemble & stacking models (R)
├── nb_20newsgroups.R                # Naive Bayes text classifier (R)
├── train_data.csv                   # Document-word training data
├── train_label.csv                  # Training labels
├── vocabulary.txt                   # Vocabulary list
└── README.md                        # Project documentation

1️⃣ Neural Network & CNN Classifier (Python)

File: CNN_NNimplementation.py

This script implements a neural network (and optionally CNN) classifier.
The workflow includes:

Data loading and preprocessing

Defining the NN/CNN model

Training loop: forward pass, loss calculation, backpropagation

Model evaluation (accuracy, loss)

Saving or printing results

Python Dependencies
pip install numpy pandas scikit-learn matplotlib
pip install torch torchvision     # If PyTorch is used
# OR
pip install tensorflow keras      # If Keras/TensorFlow is used

Run
python CNN_NNimplementation.py

2️⃣ Decision Tree Classifier (Python)

File: DTClassifier_implementation (likely .py)

This script implements:

Training a decision tree

Predicting on test data

Displaying accuracy and confusion matrix

Python Dependencies
pip install numpy pandas scikit-learn

Run
python DTClassifier_implementation

3️⃣ Ensemble Classifier With Stacking (R)

File: ensemble_classifier.R

This R script builds and evaluates multiple ML models on a binary classification dataset using the caret and caretEnsemble packages.

Models Included

Random Forest (rf)

AdaBoost (adaboost)

Logistic Regression (glm)

Decision Tree (ctree)

k-NN (knn)

Naive Bayes (nb)

Neural Network (nnet)

Workflow

Reads training and test data (lab4-train.csv, lab4-test.csv).

Converts class labels 0 → N, 1 → Y.

Trains each model with cross-validation.

Evaluates each model using confusionMatrix.

Creates an unweighted ensemble using caretList.

Builds a stacked model using caretStack with Random Forest as meta-learner.

Performs threshold tuning to minimize classification error.

Generates confusion matrices & overall accuracy.

Saves detailed output to:

Output_final_April18_2019.txt

R Dependencies
install.packages("randomForest")
install.packages("data.table")
install.packages("caret")
install.packages("caretEnsemble")
install.packages("tidyverse")      # optional but useful

Run
setwd("path/to/coms573_lab4/")
source("ensemble_classifier.R")

4️⃣ Naive Bayes Text Classifier – 20 Newsgroups (R)

File: nb_20newsgroups.R

This R script implements a multinomial Naive Bayes classifier for a subset of the 20 Newsgroups dataset.

It computes:

MLE probabilities

Bayesian (Laplace-smoothed) probabilities

Document likelihood scores (Omega_MLE, Omega_NB)

Class priors

Training-set predictions

Test-set predictions

Group-wise accuracy

Confusion matrices

Input files required

train_data.csv

Columns: document ID d, word ID w, count w_d

train_label.csv

Column: class ID n

vocabulary.txt

One word per line; script assigns IDs internally

Place these three files in your working directory.

Main Steps Performed

Merge counts and labels

Compute:

total words per document (tw_d)

total words per class (tw_n)

word-class counts (w_n)

Compute vocabulary size (vc)

Estimate:

MLE = w_n / tw_n

Bayesian = (w_n + 1) / (vc + tw_n)

Compute log-likelihoods

Compute class prior probabilities

Score each document using:

Omega_MLE = Σ log(MLE) + log(prior)
Omega_NB  = Σ log(BE)  + log(prior)


Pick the class with max Omega

Evaluate:

accuracy (overall + group-wise)

confusion matrices for MLE + Bayesian

Save results:

results_OmegaNB_trainingset.txt
results_OmegaMLE_trainingset.txt
confusionmatrix_BE_trainingset.txt
results_OmegaNB_testset.txt
results_OmegaMLE_testset.txt
confusionmatrix_MLE_testset.txt
confusionmatrix_BE_testset.txt

R Dependencies
install.packages("tidyverse")
library(tidyverse)

Run
setwd("/path/to/20newsgroups/")
source("nb_20newsgroups.R")
