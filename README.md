# Data Science **** Work in Progress *****
### GitHub Repository

This Github repository contains Sample Jupyter Notebook.
- **Data Source**
> * [Kaggle](https://www.kaggle.com/kazanova/sentiment140)

1. > * [Sentiment analysis with tweets](https://github.com/Erik1120/Profolio/blob/main/BayesianOptimization.ipynb)
    - Lancaster Stemmer
    - Count Vectorizer & Tfidf Vectorizer
    - Light GBM Classifier with Bayesian Optimization

2. 


- **A/B testing:**
> * [A/B testing](https://github.com/Erik1120/Sample/blob/main/A_B_Testing_Datacamp.ipynb)

- **Pandas:**
> * [Pandas Data Manipulation](https://github.com/Erik1120/Profolio/blob/main/Pandas/DataManipulationPandas.ipynb)

> * [Pandas Joining Data 1](https://github.com/Erik1120/Profolio/blob/main/Pandas/JoiningDataWithPandas.ipynb)

> * [Pandas Joining Data 2](https://github.com/Erik1120/Profolio/blob/main/Pandas/PandasJoinsForSpreadsheetUsers.ipynb)

> * [Reshaping data](https://github.com/Erik1120/Profolio/blob/main/Pandas/ReshapingDataUsingPandas.ipynb)







> * Word Frequency below:
![](https://github.com/Erik1120/Springboard/blob/main/Capstone/Notebook/image/words_freq.png)

### 3. Text Transformation
Tfidf & Count vectorizer:
- In the current exercise using 1.6 million tweets:
    1. Tfid required 40% additional resources to transform text data into float.
    2. Tfid and Counter vectorizer both generated 1,163 features from the text.

### 4. Automate Hyperparameter tuning
Bayesian Optimization & Keras Tuning
- Exploring two types of automated tuning for our models were explored
    1. Bayesian Optimization with Count vectorizer required a 15% increase in processing
    2. A small sample dataset for Keras tuning does not seem optimum because  AUC ranges from the worst to the best. 

### 5. Machine Learning
> * [Sampled ML Notebook](https://github.com/Erik1120/Springboard/blob/main/Capstone/Notebook/Sentiment_Data-SampleData.ipynb)

The exercise below (ML Notebook) includes multiple tree-based binary classification models, including Neural Network, to determine the best-performing per-AUC metrics. In this exercise, I am not concerned with Type 1 or Type II errors but ONLY looking at the model's accuracy. Further analysis provided for a smaller sample for Keras & MLP does not offer the best solution for our exercise. 