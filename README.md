Welcome to the Heart Attack Prediction Project! This project aims to develop a machine learning model to predict the likelihood of heart attacks based on various health indicators and behaviors. The project involves extensive Exploratory Data Analysis (EDA), data preprocessing, model training, and evaluation.

**I should note that most of the documentation is directly in the Jupyter notebooks. I documented each step quite thoroughly. Here is just a very brief summary.**

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Handling Leaky Variables](#handling-leaky-variables)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Challenges Faced](#challenges-faced)
- [Future Work](#future-work)

## Introduction

Heart disease is a leading cause of death globally. Predicting heart attacks can save lives by enabling early interventions. This project explores various machine learning techniques to predict heart attacks using health survey data from the CDC's Behavioral Risk Factor Surveillance System (BRFSS).

The task is an example of **binary, single-label classification**. The main challenges of the project were **leaking variables**, **missing values**, and the **large volume of data**.

## Dataset Description

The dataset originates from the [CDC](https://www.cdc.gov/) and is part of the [BRFSS](https://www.cdc.gov/brfss/annual_data/annual_data.htm), which conducts annual telephone surveys to collect data on U.S. residents' health-related risk behaviors, chronic health conditions, and use of preventive services.

- **Original Dataset**: Over 300 variables.
- **Current Dataset**: Reduced to 40 relevant variables.
- **Size**: Approximately 200,000 records after cleaning.
- **Source**: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

### What Does the Dataset Cover?

The dataset includes various health indicators such as:

- Demographics (Age, Sex, Race)
- Health behaviors (Smoking, Alcohol Consumption, Physical Activity)
- Medical history (Diabetes, Stroke, Mental Health)
- Physical measurements (BMI)

## Project Structure

- [heart_attack_prediction_EDA.ipynb](./heart-attack-prediction-eda.ipynb) - EDA steps and insights
- [train_models.ipynb](./train_models.ipynb) - Model training and evaluation
- [utils](./utils/)
  - [utils.py](./utils/utils.py) - Helper functions
  - [feature_encoding.py](./utils/feature_encoding.py) - Feature encoding and transformations
- [web-app](./Web-app/) - Web application (in development)

## Exploratory Data Analysis

The EDA was a most time consuming part of this project, involving:

1. Domain Research: Reviewed numerous studies to understand the relationship between each feature and heart disease. Explanation of all features and references to research within the notebook heart_attack_prediction_eda.ipynb.
2. Missing Data Analysis:

   - Identified patterns of missingness (MCAR, MAR, MNAR).
   - Built heatmaps and evaluated combinations of missing values.

3. Univariate Analysis:
   - Assessed distributions of all variables.
   - Applied transformations to continuous variables (Yeo-Johnson, logarithmic, square root).
   - Binned bimodal distributions.
4. Bivariate Analysis:

   - Evaluated relationships between features and the target variable.
   - I removed the variable RaceEthnicityCategory since the likelihood of a heart attack hardly varied among different classes. This feature could have added unnecessary noise to the data and territorial context.

   - I also removed the feature LastCheckUpTime, as the data may not be representative of its importance, even though it is indeed significant. I found that people who visit the doctor less frequently are significantly less affected by heart attacks! Of course, the issue is that if a person does not regularly visit the doctor, they do so for valid reasons. For the same reasons, they could not become respondents in the survey and thus part of the data. For example, a lack of money might lead to not having a phone, or a lack of time could result in declining to participate in the survey.
   - Identified potential data leakage in certain variables.

5. Multivariate Analysis:
   - Confirmed or refuted hypotheses from previous analyses.
   - Used statistical tests (Cramér's V) to assess feature importance.

## Handling Leaky Variables

After a series of Cramér's V statistical tests and assessing the importance of features with the baseline model, I concluded that it makes sense to remove some of the variables I suspected of leaking data.

- `HadAngina`
- `HadStroke`
- `GeneralHealth`
- `PhysicalHealthDays`
- `DifficultyWalking`

Although these variables might be good predictors, I decided to minimize the chance of data leakage, even at the cost of increasing the model's bias.

## Data Preprocessing

### Features transformation

Logarithmic transformation and standardization for continuous features with a normal distribution.

### Feature Encoding

**Ordinal Encoding**: Applied to categorical variables to prepare for imputations.
**Binning**: Continuous variables with bimodal distributions were binned to reduce complexity.
**One-hot encoding**: rest part of features

### Imputation

Due to computational constraints, advanced imputation methods like MICE and missForest were not fully utilized. The data used for modeling was the cleaned subset without missing values.

### Dimensionality Reduction

- Compared **UMAP** and **PCA** for potential feature reduction.
- Determined that encoding methods significantly impact these techniques.

## Modeling

### Baseline model

LightGBM was used as the baseline due to its speed and efficiency. Initial model trained on clean data after encoding.

### Models Comparison

Evaluated 10 different models:

- **Linear**: Logistic Regression, SVM, LDA
- **Non-linear**: Desicion tree, KNN, RandomForest, AdaBoost, XGBoost, LightGBM, CatBoost

### Findings

- **Linear Models**: Logistic Regression and LDA performed comparably to ensemble methods.
- **Ensemble Models**: Showed tendencies to overfit but had higher potential after tuning.
- **SVM and KNN**: Performed the worst and required the most computation time.

CatBoost performed the best.

### Balancing data

Tested various resampling techniques (SMOTE, Random Oversampling and Random Undersampling).

Balancing methods did not improve model performance and sometimes worsened it.

### Hyperparameter Tuning

Employed Bayesian Optimization for tuning CatBoost parameters.
Achieved a modest increase in AUC by 0.5 points.

### Feature Importance and SHAP Analysis

Analyzed feature importance to understand model decisions.
Used SHAP values to interpret the impact of each feature on predictions.

## Challenges Faced

1. Build **a clear roadmap for the project**. Seriously, there are too many stages. Moreover, they are sometimes very unclear and contradictory. To perform imputation, encoding is necessary. However, standard types of encoding cannot be used, so we need to look for alternatives. Feature encoding should be done before imputation. Feature engineering and imputation must be done after removing leaking features. It's crucial to understand the data and how it was formed based on the entire dataset, especially during training. For example, when balancing, we only oversample the training data and leave the test data untouched. And so on.

2. **The computational cost of certain tools**, especially imputation and clustering. Hyperparameter tuning is also very expensive, as is dimensionality reduction.

3. **An iterative process**. Throughout the project, I constantly found things that required me to go back and fix or complete. Because of this, the order of development was often disrupted, and I had to start maintaining a special visual map to track what I was doing and what I had already completed.

## Future Work:

- **Imputation**: Implement advanced imputation techniques to utilize the entire dataset.
- **Feature Engineering**: Explore clustering methods for feature creation despite computational costs.
- **Neural Networks**: Apply deep learning models using Keras for potential performance gains.
- **Web Application**: Finalize the web app to demonstrate model predictions interactively.
