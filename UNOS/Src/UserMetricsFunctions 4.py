import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# split test and train
from sklearn.model_selection import train_test_split
# sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
# scale
from sklearn.preprocessing import MinMaxScaler
# import lfeature selection ibrary & functions
from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold, mutual_info_classif
# import libraries for i
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical



def classifier_metrics(model, Xdata, ydata, flag=None, display=True):
    """
    Classification metric for Project includes 
    Model metrics & Confusion Matrix.
    """
    # Predictions
    pred = model.predict(Xdata)
    
    # Create confusion matrix
    cm = metrics.confusion_matrix(ydata, pred, labels=model.classes_)
    
    # Initialize variables
    TN, FP, FN, TP = cm.ravel()
    Spec = TN / (TN + FP)
    Recall = TP / (TP + FN)
    Acc = (TP + TN) / (TP + TN + FP + FN)

    if (TP + FP) == 0:
        Prec = 0  # Set precision to 0 when denominator is 0
    else:
        Prec = TP / (TP + FP)
    
    if (Prec + Recall) == 0:
        F1Score = 0  # Set F1Score to 0 when denominator is 0
    else:
        F1Score = 2 * (Prec * Recall) / (Prec + Recall)
    
    AvgPrec = metrics.average_precision_score(ydata, pred)


    if display:    
    # Print messages
        if flag:
            print("*" * 5 + " Classification Metrics for Validation/Test:")
        else:
            print("*" * 5 + " Classification Metrics for Training:")
            
        # Classification report for more metrics
        print("Classification Report:\n", metrics.classification_report(ydata, pred, zero_division=0))

    # Calculate ROC curve and AUC
    fpr, tpr, _ = metrics.roc_curve(ydata, pred)
    roc_auc = metrics.auc(fpr, tpr)
    
    if display:
        # Plot confusion matrix and ROC curve in a single figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], cbar=False, ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        if flag:
            ax1.set_title("Validation/Test Confusion Matrix")
        else:
            ax1.set_title("Training Confusion Matrix")

        # Plot ROC curve
        ax2.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax2.legend(loc="lower right")

        # Show the combined plot
        plt.tight_layout()
        plt.show()
    
    return Spec, Recall, Acc, Prec, F1Score, AvgPrec, roc_auc


def ClassificationMatric(Algorithm, Model, Desc, model, Xdata, ydata, Type, metricDF=None, display=True):
    """
    This function evaluates a classification model's performance on a given dataset by calculating 
    key metrics such as specificity, sensitivity, accuracy, precision, F1 score, average precision, 
    and AUC (Area Under the Curve). It then compiles these metrics into a DataFrame, allowing users 
    to track performance results across different models.
    """
    # determine training or validation/test
    if Type.lower() == 'training':
        flag = False
    else:
        flag = True

    # capitalize
    Type = Type.capitalize()
        
    # display report - training
    Specificity, RecallSensitivity, Accuracy, Precision, F1, AveragePrecision, AUC = classifier_metrics(model, Xdata, ydata, flag=flag, display=display)
    
    # add to DataFrame
    df_metrics = metricsClassfication(Algorithm, Model, Desc, Type, Specificity, RecallSensitivity, Accuracy, Precision, F1, AveragePrecision, AUC)

    # check existing DataFrame
    if metricDF is not None and not metricDF.empty:
        # concat two dataframes
        dfNew = pd.concat([metricDF, df_metrics], ignore_index=True)

        # reset the index
        dfNew.reset_index(drop=True, inplace=True)
    else:
        # copy first metrics dataframe
        dfNew = df_metrics.copy()

    return dfNew 



def stratified_grid(model, parameters, Xdata, ydata, seed, nJobs=-1, nSplit=5, score = 'roc_auc'):
    """
    Ten fold CV Stratified
    """
    # instantiate Stratified K-Fold cross-validation takes into account the class distribution
    cv = StratifiedKFold(n_splits=nSplit, shuffle=True, random_state=seed)

    # perform GridSearchCV
    GSC_estimator = GridSearchCV(model, parameters, scoring=score, cv=cv, n_jobs=nJobs)

    # evaluate a score by cross-validation
    scores = cross_val_score(GSC_estimator, X=Xdata, y=ydata, scoring=score, cv=cv, n_jobs=nJobs)

    # print average accuracy score CV with standard deviation
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    # fit model
    fit = GSC_estimator.fit(Xdata, ydata)
    
    return fit


def LogisticFeatureImportance(model, figsize=(8,10), fontsize=8):
    """
    This function analyzes the importance of features in a logistic regression model by processing its 
    coefficients. It creates a DataFrame with each feature's name, coefficient, effect description, 
    odds ratio, percentage change in odds, and probability, including a barh plot.
    """
    # determine feature information
    feature_names = model.feature_names_in_
    coefficients = model.coef_
    
    # create a DataFrame
    LRcoeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients[0],
        'Description': ['Decrease in the log-odds of the Positive Class' if x < 0 else 'Increase in the log-odds of the Positive Class' for x in coefficients[0]],
        'Odd Ratio': np.exp(coefficients[0]),
        'Percentage Change in Odds': (np.exp(coefficients[0]) - 1) * 100,
        'Probability': np.exp(coefficients[0]) / (1 + np.exp(coefficients[0]))
    })

    # sort by Coefficient
    LRcoeff_df = LRcoeff_df.sort_values(by='Coefficient')

    # reset the index
    LRcoeff_df.reset_index(drop=True, inplace=True)

    # plot feature importance
    LRcoeff_df.plot(kind='barh', x='Feature', y='Coefficient', figsize=figsize, title="Feature Importance (Logistic Regression)", fontsize=fontsize)
    plt.axvline(0, color='red', linestyle='-')
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Features")
    plt.show()

    return LRcoeff_df 


def plotFeatureImportance(model, Xdata, figsize=(30,30), display=True):
    """
    Plot feature importance from the model
    Order List & Bar Plot of Importance
    """
    # create dataframe
    data = pd.DataFrame(model.feature_importances_ , index=Xdata.columns, columns=["Feature Importance Score"])
    # print(data.sort_values("% Feature Importance", axis=0, ascending=False))

    if display:
    # bar plot
        plt.figure(figsize=figsize)
        # create a bar plot using Seaborn
        ax = sns.barplot(data=data, y=data.index, x = data['Feature Importance Score'], orient= 'h')
        ax.set_title("Feature Importance Bar Plot", fontsize = 15)
        # add a grid to the x-axis/
        plt.grid(axis='x', linestyle='--')
        plt.show()

    return data



def metricsClassfication(Algorithm, Model, Desc, Type, S, R, A, P, F, AP, Auc):
    """
    Pass Classfication metrics and Model Information
    """
    # initialize DataFrame
    data = pd.DataFrame(columns=['Algorithm', 'Model', 'Description', 'DataType', 'Accuracy', 'RecallSensitivity','F1Score', 'AveragePrecision', 'Precision','Specificity', 'ROC_AUC_Score'])
    # write to DataFrame
    data.loc[len(data)] = [Algorithm, Model, Desc, Type, A, R, F, AP, P, S, Auc]

    return data


def metricsClassifier(model, Xdata, ydata, data, flag='Train'):
    """
    The metricsClassifier function calculates classification metrics for a 
    given model and appends them to an existing DataFrame.
    """
    # initialize variable
    Type = flag
    
    if Type == 'Train':
        Test = False
    else:
        Test = True
    
    # display report - training
    S, R, A, P, F, AP, Auc = classifier_metrics(model, Xdata, ydata, Test)
        
    # add to DataFrame
    df_metrics = metricsClassfication(Algorithm, Model, Desc, Type, S, R, A, P, F, AP, Auc)
    
    # concat two dataframes
    data = pd.concat([data, df_metrics], ignore_index=True)
    
    # reset the index
    data.reset_index(drop=True, inplace=True)
    
    return data