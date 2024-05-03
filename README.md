<p align="center">
  <img src="UTA-DataScience-Logo.png" />
</p>

<!-- Use this div with the custom class around your text -->
<div class="custom-text">

## Kaggle Challenge: Mushroom Classification
This project explores mushroom classification based off of edibility versus poisonous using machine learning techniques on the Mushroom Classification dataset from the UCI Machine Learning Repository (https://www.kaggle.com/datasets/uciml/mushroom-classification).


## Overview

The primary task involves classifying mushrooms into edible or poisonous categories based on their features, as defined by the Mushroom Classification dataset from the UCI Machine Learning Repository.

My approach frames the problem as a binary classification task, utilizing various machine learning algorithms to predict the mushroom's class.

I used XGBoost and KNN models to assess their performance. Ultimately, the best-performing model achieved an accuracy of 99%, indicating strong predictive capability on this dataset.

## Summary of Workdone

### Data

Type: Tabular data (CSV)

Input: Mushroom features encoded as categorical variables

Output: Classification label indicating whether the mushroom is edible or poisonous

Size: 8124 instances, 23 features (excluding the target variable)

Instances:

Training set: 6499 instances
Testing set: 1625 instances
Validation set: Not specified

#### Preprocessing / Clean up

* Exploratory data analysis, data preprocessing, data cleaning, data visualization, and feature engineering on training and test data

#### Data Visualization
<p>

A perfect AUC of 1 and a flat ROC curve probably means there is overfitting in XGBoost. Precision, accuracy, and F1 scores are all 1, meaning the model isn't making correct predictions on testing data.

Potential reasons for XGBoost overfitting:

Model Complexity: XGBoost can capture complex patterns, leading to overfitting if the model is too complex for the dataset size.
Hyperparameters: Improper tuning of hyperparameters like tree depth or learning rate can lead to overfitting.
Data Size and Quality: Small or unrepresentative datasets can cause overfitting.
The correlation matrix probably was hard to read due to an encoding error.
</p>


### Problem Formulation


Machine learning can play a crucial role in the accurate classification of mushrooms, which would be a great help in food safety and identification. The challenge is to develop robust models that can effectively differentiate between edible and poisonous mushrooms based on their features.

# Pretrained ML Model for Mushroom Classification:
In this context, I pre-trained two machine learning models: XGBoost and KNN. These models were selected for their feature selection and accuracy in handling classification tasks and their potential to provide  predictions for mushroom types.

Models:
I used two final models that demonstrated promising performance in mushroom classification:

### Training

  * Training was done in Google Colab on a Dell XPS laptop.
  * A lot of my issues were due to some confusion I had, as I tried both label and one hot encoding and it took a while to figure out why my code kept tossing up errors, whether it not finding 'class' or axis. The ROC curve is less of a curve and more a line, meaning my model is probably not predicting well. I'm unsure of why the correlation matrix turned out like this. Also sorry for using so much seaborn, it's what Im most comfortable with.
<p><div><img src="" width="600" height="500"></div></p>

### Performance Comparison

* XGBoost did not perform well in terms of overfitting, and the KNN value was found the square root of instances- it is an odd number, but it's 75, which I think Is a bit high.

#### ROC Curve Graph

<p><div><img src="![image](https://github.com/thaoitha/Project3402/assets/113535597/238d24f5-6f86-4176-9c28-9b7573991f6e)
" width="600" height="500"></div></p>

### Conclusions

* The KNN model probably did the best, considering the scores and metrics.

### Future Work

*For future work, I will probably spend more time hypertuning the KNN model
  
## How to reproduce results

* To replicate my exact results:
   * Use colab, download dataset from Kaggle, Perform preprocessing, one hot and label encoding, 
   *

### Overview of files in repository

  * The repository has 3 files
  * 
  * README.md: Breakdown of Github repository and files.
  



### Software Setup

* Software libraries and packages needed: Scikit-Learn, Numpy, Seaborn, Pandas, Matplotlib, Math, XGBoost, IPython, and tabulate.
* From the libraries you can import the specific packages listed at the top of each notebook that you will need. If your machine does not have it check online. Most if not all of them have documentation for installing on your machine.


### Data

* The original training and testing data can be downloaded from the Kaggle link above. You can download it from there.

### Training

* The most imporant thing I learned was that cross validation is necessary, and to use more graphs to visualize how well the model was trained.

#### Performance Evaluation

* The models did not perform well, but KNN performed marginally better.
* Root Mean Squared Error: 0.028452396799091593 for KNN

## Citations & Acknowledgements
* Thank you to Amir Farbin for his wonderful lectures
* https://www.kaggle.com/datasets/uciml/mushroom-classification

</div>






