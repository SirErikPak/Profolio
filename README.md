# Data Science 
### GitHub Repository

This Github repository contains Sample Jupyter Notebook.
- **Data Source**
> * [Kaggle](https://www.kaggle.com/kazanova/sentiment140)

1. > * [Sentiment analysis with tweets](https://github.com/Erik1120/Profolio/blob/main/BayesianOptimization.ipynb)
    - Lancaster Stemmer
    - Count Vectorizer & Tfidf Vectorizer
    - Light GBM Classifier with Bayesian Optimization

2. 
- **A/B testing Notes:**
> * [A/B testing](https://github.com/Erik1120/Sample/blob/main/A_B_Testing_Datacamp.ipynb)

- **Pandas Data Manipulation:**
> * [Pandas Data Manipulation](https://github.com/Erik1120/Profolio/blob/main/Pandas/DataManipulationPandas.ipynb)


### 2. Data Cleaning

> * [Data Cleaning Notebook](https://github.com/Erik1120/Springboard/blob/main/Capstone/Notebook/SentimentAnalysis_wrangling.ipynb)

1. Applied Regular expression operations to remove noise from the tweets. Also, Lancaster Stemmer was used to determining the common root form and from nltk.corpus to generate stopwords removal for our sentiment analysis data.

2. All the features except target and text were remove from our analysis.





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


> * [ML Notebook](http://localhost:8888/notebooks/Springboard/Capstone/Notebook/Sentiment_Data.ipynb)

#### Algorithm:

- In this exercise, I have generated Keras Classifier, MLP Classifier, Logistic Regression, and LGBM Classifier models to examine the best performing classifier between these models. Also, used Bayesian Optimization for LGBM Classifier for hyperparameter tuning, including Keras Tuning for Keras and MLP Classifier to determine the hidden layer for both models.
    1. The count vectorizer attained a higher AUC score for MLP and Keras, which are Neural networks. In Logistic Regression and  LGBM Classifier, the Tfidf vectorizer generated higher AUC scores. 
    2. Logistic Regression seemed to be an excellent algorithm for our classification exercise, and using Bayesian Optimization with LGBM did NOT outperform Logistic Regression.
    3. Keras & MLP had the highest scores, but these models would require additional tuning because training and test scores are much further apart from each other, especially in a sampled dataset. So this clue states that our models are memorizing our test dataset, especially in our sampled dataset.
    
    
#### Full Data / Sampled AUC Performance:
> * Performance full Log below:
![](https://github.com/Erik1120/Springboard/blob/main/Capstone/Notebook/image/log_styled.png)
>
> * Performance sample Log below:
![](https://github.com/Erik1120/Springboard/blob/main/Capstone/Notebook/image/log_styled_sample.png)
    
