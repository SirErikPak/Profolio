# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
# import library
import scipy.stats as stats
from scipy.stats import chi2_contingency
from itertools import combinations
from scipy.stats import mannwhitneyu
# split test and train
from sklearn.model_selection import train_test_split
# sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
# scale
from sklearn.preprocessing import MinMaxScaler
# import lfeature selection ibrary & functions
from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold, mutual_info_classif

# import libraries for BayesianO ptimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# set seed
RANDOM_STATE = 1776


def insertIntoDataFrame(data, values):
    """
    This function inserts each value from 
    the list values as a new row in the 
    DataFrame data.
    """
    for value in values:
        data.loc[len(data)] = value
    return data


def mannwhitneyu_combinations(data, features):
    """
    The function performs the Mann-Whitney U test on all combinations of two features 
    in a given list of features. The test checks if there is a significant difference 
    between the distributions of two independent variables. It prints and returns the
    U-statistic and p-value for each pair of features.
    """
    results = []  # store the results if you want to return them
    
    # iterate over all combinations of features
    for feature1, feature2 in combinations(features, 2):
        # extract the data for the two features
        group1 = data[feature1].dropna()  # remove NaN values
        group2 = data[feature2].dropna()  # remove NaN values
        
        # perform the Mann-Whitney U test
        U_statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # store the result
        result = (feature1, feature2, U_statistic, p_value)
        results.append(result)
        
        # Display the result
        print(f"Features: {result[0]} vs {result[1]}, U-statistic: {result[2]:,}, P-value: {result[3]:.5f}")
    
    return results



def combineGetUnique(data, col1, col2, nanVal, flag=True):
    """
    This function combines unique values from two specified columns of a DataFrame, 
    replaces NaN values with a user-defined value, and removes duplicates to return a 
    list of unique values. It also allows handling different data types for missing 
    values (NaN for floats and <NA> for integers).
    """
    # determine uniquew values
    check1, check2 = data[col1].unique(), data[col2].unique()
    if flag:
        # replace NaN with nanVal for float datatype (nan)
        check1 = np.where(np.isnan(check1), nanVal, check1).astype(int)
        check2 = np.where(np.isnan(check2), nanVal, check2).astype(int)
    else:
        # replace NaN with nanVal for int datatype <NA>
        check1 = pd.Series(check1).fillna(nanVal).astype(int).values
        check2 = pd.Series(check2).fillna(nanVal).astype(int).values
        
    # concatenate two array
    check = np.concatenate((check1, check2))
    # remove dups
    check = set(check)

    return list(check)


def getUnique(data, col1, nanVal, col2=None, flag=True):
    """
    This function extracts unique values from one or two columns in a DataFrame, 
    optionally replacing missing values with a specified value, and returns the 
    results as a deduplicated list. It provides flexibility for handling 
    float (NaN) and integer (<NA>) missing values based on the data type.
    """
    # determine uniquew values
    check1 = data[col1].unique()
    
    if flag:
        # replace NaN with nanVal for float datatype (nan)
        check1 = np.where(np.isnan(check1), nanVal, check1).astype(int)
    else:
        # replace NaN with nanVal for int datatype <NA>
        check1 = pd.Series(check1).fillna(nanVal).astype(int).values

    if col2:
        # concatenate two array
        check = np.concatenate((check1, col2))
        # remove dups
        check = set(check)
    else:
        check = set(check1)
        

    return list(check)


def toCategory(data, col):
    """
    This function converts specified columns in a DataFrame to the category data type.
    """
    # iterate
    for val in col:
        data[val] = data[val].astype('category') 

    return data


def mappingCol(data, colstr, mapdict, flag=True):
    """
    This function is designed to map values in a specified column of a DataFrame to new values 
    using a provided dictionary and optionally print the unique values after the mapping and converts 
    the column to the category data type.
    """
    # map values in the column using the mapping dictionary, leaving values unchanged if not found in the mapping
    data[colstr] = data[colstr].map(mapdict).fillna(data[colstr])
    # convert datatype to categorical data type
    data[colstr] = data[colstr].astype('category')

    if flag:
        print(f"Converted Column {colstr} Unique Vaue(s) {data[colstr].unique()}")

    return data


def colInfo(data, datadict, column, flag=False):
    """
    Return column(s) and display descriptive statistics/NaN count/Datatypes, 
    including displaying unique records.
    """
    # feature name
    feature = datadict.featureName[datadict.featureName.str.contains(column)].values

    if len(feature) == 0:
        raise ValueError(f"No matching features found for '{column}'")

    
    # descriptive statistics
    print(data[feature].describe(include='all').T)
    
    # check NaN & datatype
    print('\n')
    print(data[feature].isna().sum())
    print(data[feature].dtypes)
    print('\n')
    
    # iterate over feature if flag TRUE
    if flag:
        for i in feature:
            print(f"{i}: {data[i].unique()}")

    return feature


def catNumeric(data, catCols):
    """
    This function checks which columns, from a list of categorical columns, contain 
    entirely numeric values when interpreted as strings.
    """
    # initialized variable
    numericValues =[]
    nonNumericValues =[]
    # check categorical columns for numeric values
    for col in catCols:
        series = pd.Series(data[col].astype(str))
        # convert to numeric, coercing errors to NaN to determine numerical category variable
        if pd.to_numeric(series, errors='coerce').notna().all():
            numericValues.append(col)
        else:
            nonNumericValues.append(col)
    
    # return
    return numericValues, nonNumericValues


def percentageNull(datadf):
    """
    This function calculates the percentage of missing values (NaN) for each column in a DataFrame 
    and returns those columns that contain any missing values, along with their corresponding NaN 
    count and percentage.
    """
    # calculate the percentage of non-null values for each column
    per_calc = pd.DataFrame(100 - (datadf.count() / len(datadf) * 100))
    
    # rename columns name
    per_calc.rename(columns={0: 'percentage'}, inplace=True)

    # add counter
    per_calc['NaNCount'] = datadf.isna().sum()
    
    # sort
    per_calc.sort_values(by='percentage', inplace=True, ascending=False)

    # 
    NanReturn = per_calc[per_calc.NaNCount != 0]
    
    return NanReturn


def find_duplicates(lst):
    return list(set([item for item in lst if lst.count(item) > 1]))



def intersectionCol(datadict, cat, string, flag=True):
    # initialize variable
    parm =  "r'(?i)" + string + "'" # regex search using ignore case sensitivity
    parm = eval(parm)
    
    # regex search
    catColD = datadict.featureName[datadict.dataType.str.contains(parm)].values
    print("Features from Data Dictionary: ", len(catColD))
    catCol = set(cat).intersection(set(catColD))
    print("Intersection Features: ", len(catCol))

    # display
    missingCol = set(catColD) - set(catCol)
    print("Missing Count", len(missingCol))

    if not flag:
        return list(catCol), list(missingCol)
    


def DefinitionSearch(datadic, col, flag=False):
    """
    This function is designed to search for a given column name (col) in a DataFrame (datadic) 
    based on the featureName column. It uses regular expression (regex) to perform a case-insensitive 
    search and returns a subset of the DataFrame or a list of feature names, depending on the flag parameter.
    """
    # initialize variable
    parm =  "r'(?i)" + col + "'" # regex search using ignore case sensitivity
    parm = eval(parm)
    # display
    df_str = datadic.loc[:,['featureName','desc', 'dataType', 'labelSAS', 'COMMENT', 'Information']][datadic.featureName.str.contains(parm) & \
                ~datadic.Information.str.contains('DROP', case=False, na=False)]

    if flag:
        feature = datadic.featureName[datadic.featureName.str.contains(parm) & ~datadic.Information.str.contains('DROP', case=False, na=False)].tolist()
        return feature
    else:
        return df_str
    
    
   
def removeColumn(datadf, col, display=True):
    """
    Remove unwanted columns
    """
    if display:
        # display removed feature(s)
        print(f"\nRemoved Features:{sorted(col)}\n")
        # display shape of DataFrame
        print(f"Total rows before: {datadf.shape[0]:,} & columns: {datadf.shape[1]:,}")
    
    # remove column
    datadf = datadf.drop(columns=col, axis=1)

    # reset index in place
    datadf = datadf.reset_index(drop=True)

    if display:
        # display shape of DataFrame
        print(f"Total rows after: {datadf.shape[0]:,} & columns: {datadf.shape[1]:,}")

    return datadf


def removeRowUsingMask(datadf, removeColLst, colstr, display=True):
    # boolean mask
    mask = ~datadf[colstr].isin(removeColLst)
    
    # apply the mask to keep only rows where 'removeColLst'
    datadf = datadf[mask]
    
    # reset the index if needed
    datadf = datadf.reset_index(drop=True)

    if display:
        # disply row removed msg
        print(f"Remove row(s) from df_{colstr} DataFrame.")

    return datadf


def updateDataDict(data, remove, txt= '', col="COMMENT", display=True):
    """
    Maintain data dictionary
    """
    # update data dictionary
    idx = data[data.featureName.isin(remove)].index
    # append to exiting data
    data.loc[idx,col] = f"**DROP** {txt} - " + data[col]

    if display:
        # disply update msg
        print(f"Data Dictionary Updated.")

    return data


def HouseKeeping(data, removeColLst, dataDict, dataRemove, dataLabel, dataCAN, dataDON, dataBoth, dataOrdinal, dataNominal, txt, display=True):
    """
    Run helper fuction for house keeping
    """
    # update data dictionary (house keeping)
    dataDict = updateDataDict(dataDict, removeColLst, txt, display=display)
    
    # remove DataFrame data (house keeping)
    dataRemove = removeRowUsingMask(dataRemove, removeColLst, colstr='remove', display=display)
    dataLabel = removeRowUsingMask(dataLabel, removeColLst, colstr='label', display=display)
    dataCAN = removeRowUsingMask(dataCAN, removeColLst, colstr='can', display=display)
    dataDON = removeRowUsingMask(dataDON, removeColLst, colstr='don', display=display)
    dataBoth = removeRowUsingMask(dataBoth, removeColLst, colstr='both', display=display)
    dataOrdinal = removeRowUsingMask(dataOrdinal, removeColLst, colstr='ordinal', display=display)
    dataNominal = removeRowUsingMask(dataNominal, removeColLst, colstr='nominal', display=display)
    
    # remove features
    data = removeColumn(data, removeColLst, display=display)


    return data, dataDict, dataRemove, dataLabel, dataCAN, dataDON, dataBoth, dataOrdinal,  dataNominal


def removeHouseKeeping(data, removeColLst, dataBool, dataOrdinal, dataNominal, dataNumeric):
    """
    Run helper fuction for house keeping
    """
    # remove DataFrame data (house keeping)
    dataBool = removeRowUsingMask(dataBool, removeColLst, colstr='boolean')
    dataOrdinal = removeRowUsingMask(dataOrdinal, removeColLst, colstr='ordinal')
    dataNominal = removeRowUsingMask(dataNominal, removeColLst, colstr='nominal')
    dataNumeric = removeRowUsingMask(dataNumeric, removeColLst, colstr='numeric')
    
    # remove features
    data = removeColumn(data, removeColLst)

    return data, dataBool, dataOrdinal, dataNominal, dataNumeric


def datatypeDF(data, display=True):
    # initialize variables for all the column name per each datatype
    boolCol = data.select_dtypes(include=['bool']).columns.tolist()
    catCol = data.select_dtypes(include=['category']).columns.tolist()
    objCol = data.select_dtypes(include=['object']).columns.tolist()
    numCol = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    otherCol = data.select_dtypes(exclude=['bool', 'category', 'object', 'int64', 'float64']).columns.tolist()

    if display:
        # display feature counts
        print('Total Data feature count: ', data.shape[1])
        print(f"\nBoolean feature count: {len(boolCol)}")
        print(f"Category feature count: {len(catCol)}")
        print(f"Numeric feature count: {len(numCol)}")
        print(f"Object feature count: {len(objCol)}")
        print(f"Other feature count: {len(otherCol)}\n")
        print('Total feature count: ' ,len(boolCol) + len(catCol) + len(objCol) + len(numCol) + len(otherCol))
    else:
        return boolCol, catCol, objCol, numCol, otherCol


def addtionalInfo(data, lst):
    """
    This function provides a concise summary for each column in a given list, displaying key descriptive statistics.
    """
    # iterate
    for val in lst:
        # mode (first if multiple)
        modeValue = data[val].mode()[0]
        modePercentage = data[val].value_counts(normalize=True, dropna=False)[modeValue]
        modeCount = data[val].value_counts()[modeValue]
        unique = data[val].nunique(dropna=False)

        # display
        print(f"**{val}** Unique: {unique} & Mode: {modeValue} & Occurrence Count: {modeCount:,} & Percentage Occurrence: {(modePercentage * 100):.2f}%")


def SpearmanRankCorrelation(data, col1, col2):
    """
    Spearman's rank correlation is a non-parametric measure of the monotonicity of the relationship between two variables.
    """
    data = data.dropna()
    correlation, p_value = stats.spearmanr(data[col1], data[col2])
    r, p = stats.pearsonr(data[col1], data[col2])

    print(f"Spearman correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}\n")
    print(f"Pearson correlation coefficient: {r:.4f}")
    print(f"P-value: {p:.4f}")


def removeCatZeroCount(data):
    """
    Remove category with no category values
    """
    # iterate each categorical column
    for column in data.select_dtypes(['category']).columns:
        # get counts of each category
        category_counts = data[column].value_counts()
        
        # remove categories with zero counts
        categories_to_keep = category_counts[category_counts > 0].index
        data[column] = data[column].cat.remove_categories([cat for cat in data[column].cat.categories if cat not in categories_to_keep])

    return data


def plotCount(data, cols, width=16, height=10, angle=False):
    # figure size
    plt.figure(figsize=(width, height))
    
    # assuming cols is your list of categorical columns
    for i, col in enumerate(cols, start=1):
        plt.subplot(3, 4, i)
        sns.countplot(data=data, x=col)
        plt.title(f'Countplot of {col}')
        plt.xlabel(col)
        if angle:
            plt.xticks(rotation=90) 
    
    plt.tight_layout()  # adjust the layout
    plt.show()


def pairColsMultiIndependenceCat(data, catCol):
    """
    Hypothesis testing using Chi-square statistic and calculating Cramer's V to 
    build consensus for all the categorical variables. 
    """
    # initialize variables
    results = []
    validCols = [col for col in catCol if col in data.columns]

    # use combinations to get unique pairs of columns
    for col1, col2 in combinations(validCols, 2):
        # create a contingency table
        contingencyTable = pd.crosstab(data[col1], data[col2])
        chi2, p_value, _, _ = chi2_contingency(contingencyTable)
        # total number of observations
        n = contingencyTable.values.sum()
        # get the number of categories in each variable (rows and columns)
        r, k = contingencyTable.shape
        min_dim = min(r-1, k-1)
        
        # handle division by zero
        if n * min_dim == 0:
            cramer_v = np.nan
        else:
            cramer_v = np.sqrt(chi2 / (n * min_dim))
        
        results.append({
            'column1': col1,
            'column2': col2,
            'chi2': chi2,
            'p_value': p_value,
            'cramer_v': cramer_v
        })
    
    return pd.DataFrame(results).sort_values(by='cramer_v', ascending=False)
    
    
def testIndependenceCat(data, cat1, cat2, flag=False):
    """
    Hypothesis testing using Ch-square statistic and calculating Cramer's V to 
    build consensus for two categorical variables.
    """
    # create a contingency table
    contingencyTable = pd.crosstab(data[cat1], data[cat2])
    # perform Chi-square test
    chi2, p_value, _, _ = chi2_contingency(contingencyTable)
    
    # total number of observations
    n = contingencyTable.sum().sum()
    # get the number of categories in each variable (rows and columns)
    r, k = contingencyTable.shape
    # calculate Cramer's V
    cramer_v = np.sqrt(chi2 / (n * min(k-1, r-1)))

    # display
    print(f"Test of Independence for Catergorical Variables: {cat1} & {cat2}")
    print(f"Chi-square statistic: {chi2:,.2f}")
    print(f"p-value: {p_value:,.4f}")
    print(f"Cramer's V: {cramer_v:,.4f}")

    if flag:
        return contingencyTable

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

# def classifier_metrics(model, Xdata, ydata, flag = None):
#     """
#     Classfication metric for Project incldues 
#     Model metrics & Confusion Matrix.
#     """
#     # predictions
#     pred = model.predict(Xdata)
    
#     # create confusion matrix
#     cm = metrics.confusion_matrix(ydata, pred, labels=model.classes_)
    
#     # initialize variable
#     TN, FP, FN, TP = cm.ravel()
#     Spec = TN / (TN + FP)
#     Recall = TP / (TP + FN)
#     Acc = (TP + TN) / (TP + TN + FP + FN)

#     if (TP + FP) == 0:
#         Prec = 0  # Set precision to 0 when denominator is 0
#     else:
#         Prec = TP / (TP + FP)
    
#     # Prec = TP / (TP + FP)
#     if (Prec + Recall) == 0:
#         F1Score = 0  # Set F1Score to 0 when denominator is 0
#     else:
#         F1Score = 2 * (Prec * Recall) / (Prec + Recall)
#     # F1Score = 2 * (Prec * Recall) / (Prec + Recall)
#     AvgPrec = metrics.average_precision_score(ydata,pred)
        
#     # print msgs
#     if flag:
#         print("*" * 5 + " Classfication Metrics for Validation/Test:")
#     else:
#         print("*" * 5 + " Classfication Metrics for Training:")
        
#     # classification report for more metrics
#     print("Classification Report:\n", metrics.classification_report(ydata, pred, zero_division=0))

#     # create the ConfusionMatrixDisplay with labels
#     cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels = model.classes_)

#     # plot the confusion matrix
#     cm_display.plot(cmap='Blues', values_format='d', colorbar=False)
    
#     #
#     if flag:
#         cm_display.ax_.set_title("Validation/Test Confusion Matrix")
#     else:
#         cm_display.ax_.set_title("Training Confusion Matrix")

#     # remove grid from plot
#     plt.grid(False)
#     # plot
#     plt.show()
#     # spacing
#     print("\n")

#     # calculate ROC curve and AUC
#     fpr, tpr, _ = metrics.roc_curve(ydata, pred)
#     roc_auc = metrics.auc(fpr, tpr)
    
#     # plot ROC curve
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.show()
    
#     return Spec, Recall, Acc, Prec, F1Score, AvgPrec, roc_auc


def stratified_grid(model, parameters, Xdata, ydata, nJobs=-1, nSplit=5, score = 'roc_auc', seed=RANDOM_STATE):
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


def plotFeatureImportance(model, Xdata, figsize=(30,30), display=True):
    """
    Plot feature importance from the model
    Order List & Bar Plot of Importance
    """
    # create dataframe
    data = pd.DataFrame(model.feature_importances_ * 100, index=Xdata.columns, columns=["% Feature Importance"])
    # print(data.sort_values("% Feature Importance", axis=0, ascending=False))

    if display:
    # bar plot
        plt.figure(figsize=figsize)
        # create a bar plot using Seaborn
        ax = sns.barplot(data=data, y=data.index, x = data['% Feature Importance'], orient= 'h')
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


# def metricsClassifier(model, Xdata, ydata, data, flag='Train'):
#     """
#     The metricsClassifier function calculates classification metrics for a 
#     given model and appends them to an existing DataFrame.
#     """
#     # initialize variable
#     Type = flag
    
#     if Type == 'Train':
#         Test = False
#     else:
#         Test = True
    
#     # display report - training
#     S, R, A, P, F, AP, Auc = classifier_metrics(model, Xdata, ydata, Test)
        
#     # add to DataFrame
#     df_metrics = metricsClassfication(Algorithm, Model, Desc, Type, S, R, A, P, F, AP, Auc)
    
#     # concat two dataframes
#     data = pd.concat([data, df_metrics], ignore_index=True)
    
#     # reset the index
#     data.reset_index(drop=True, inplace=True)
    
#     return data


def corrCols(df, method='pearson', threshold=0.9, flag=False):
    """
    This function is designed to identify pairs of features that are highly correlated in a dataset. 
    It calculates the correlation matrix of numerical columns and identifies pairs of features where 
    the absolute correlation is greater than a given threshold (default is 0.9).
    """
    # initilaize variable
    feature = list()
    # calculate the correlation matrix
    correlation_matrix = df.select_dtypes(exclude='object').corr(method=method)
    
    # get the number of features
    num_features = correlation_matrix.shape[0]
    
    # iterate over the upper triangular part of the matrix
    for i in range(num_features):
        for j in range(i+1, num_features):
            feature1 = correlation_matrix.index[i]
            feature2 = correlation_matrix.columns[j]
            correlation = correlation_matrix.iloc[i, j]
            if abs(correlation) > threshold:
                feature.append(feature2)
                print(f"Correlation between {feature1} and {feature2}: {correlation:.3f}")

    if flag:
        return feature
    else:
        return


def EncodeDummyTrainValTest(data, labelTxt, nominalColumns, seed):
    """
    This function performs dummy encoding on nominal columns, splits the dataset into training, 
    validation, and test sets, and returns the processed datasets. It ensures that the label column 
    is excluded from the nominal columns to prevent encoding the target variable.
    """
    # remove label column from nominalColumns if it exists
    if labelTxt in nominalColumns:
        # remove label
        nominalColumns.remove(labelTxt)

    # dummy Encoding
    df_encoded = pd.get_dummies(data, columns=nominalColumns, drop_first=True, dtype=int)

    # entire features
    X = df_encoded.drop(labelTxt, axis=1)
    y = df_encoded[labelTxt]
    
    # split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    # split train data into validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)
    
    # display shape
    print(f"Training Dependent Shape: {X_train.shape} & Label Shape: {y_train.shape}")
    print(f"Validation Dependent Shape: {X_val.shape} & Label Shape: {y_val.shape}")
    print(f"Testing Dependent Shape: {X_test.shape} & Label Shape: {y_test.shape}")

    return  X, y, X_train, X_test, X_val, y_train, y_val, y_test


def EncodeDummyScaleAllTrainValTest(data, labelTxt, nominalColumns, feature_range = (0, 1), vflag=True, seed=RANDOM_STATE):
    """
    This function prepares a dataset for machine learning by performing dummy encoding on nominal columns, 
    scaling all features, and splitting the data into training, validation, and test sets. It can return the 
    scaled features as either numpy arrays or pandas DataFrames, based on the flag parameter.
    """
    # remove label column from nominalColumns if it exists
    if labelTxt in nominalColumns:
        # remove label
        nominalColumns.remove(labelTxt)

    # dummy Encoding
    df_encoded = pd.get_dummies(data, columns=nominalColumns, drop_first=True, dtype=int)

    # entire features
    X = df_encoded.drop(labelTxt, axis=1)
    y = df_encoded[labelTxt]
    
    # split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    # split train data into validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)

    # initialize scaling
    scaler = MinMaxScaler(feature_range=feature_range)

    # fit model
    fit = scaler.fit(X_train)

    # transform
    X_train = fit.transform(X_train)
    X_val = fit.transform(X_val)
    X_test = fit.transform(X_test)

    # if flag:
    #     # convert to dataframe
    #     X_train = pd.DataFrame(X_train, columns=X.columns)
    #     X_val = pd.DataFrame(X_val, columns=X.columns)
    #     X_test = pd.DataFrame(X_test, columns=X.columns)
    
    # display shape
    print(f"Training Dependent Shape: {X_train.shape} & Label Shape: {y_train.shape}")
    print(f"Validation Dependent Shape: {X_val.shape} & Label Shape: {y_val.shape}")
    print(f"Testing Dependent Shape: {X_test.shape} & Label Shape: {y_test.shape}")

    return  X, y, X_train, X_test, X_val, y_train, y_val, y_test


def EncodeDummyScaleTrainValTest(Xdata, ydata, nominalColumns, numericColumns, feature_range, seed=RANDOM_STATE):
    """
    This function preprocesses a dataset for machine learning by encoding nominal variables, scaling numeric features, 
    and splitting the dataset into training, validation, and test sets. It ensures all steps, from encoding to scaling,
     are applied consistently and provides stratified sampling based on a specified target class.
    """
    # dummy Encoding
    df_encoded = pd.get_dummies(Xdata, columns=nominalColumns, drop_first=True, dtype=int)
    
    # split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(df_encoded, ydata, test_size=0.2, random_state=seed, stratify=ydata['status'])
    
    # split train data into validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train['status'])

    # initialize scaling
    scaler = MinMaxScaler(feature_range=feature_range)

    # fit model
    fit = scaler.fit(X_train[numericColumns])

    # transform
    X_train[numericColumns] = fit.transform(X_train[numericColumns])
    X_val[numericColumns] = fit.transform(X_val[numericColumns])
    X_test[numericColumns] = fit.transform(X_test[numericColumns])
    
    # display shape
    print(f"Training Dependent Shape: {X_train.shape} & Label Shape: {y_train.shape}")
    print(f"Validation Dependent Shape: {X_val.shape} & Label Shape: {y_val.shape}")
    print(f"Testing Dependent Shape: {X_test.shape} & Label Shape: {y_test.shape}")

    # return  X, y, X_train, X_test, X_val, y_train, y_val, y_test
    return  X_train, X_test, X_val, y_train, y_val, y_test

def selectKClassif(Xdata, ydata, K='all', function='f_classif'):
    """
    This function selects the most relevant features for classification tasks using statistical tests provided 
    by SelectKBest and removes constant features from the dataset. It also returns detailed information on the 
    chosen features, including their scores and p-values.
    """
    # remove constant features
    constanFilter = VarianceThreshold(threshold=0)  # removes features with zero variance
    X_data = constanFilter.fit_transform(Xdata)

    # update feature names after removing constant features
    remaining_feature_names = Xdata.columns[constanFilter.get_support(indices=True)]
    
    # Step 3: Apply SelectKBest with F-classif
    selector = SelectKBest(score_func=eval(function), k=K)
    X_new = selector.fit_transform(X_data, ydata)

    # update feature names to reflect remaining, selected features
    selected_List = [remaining_feature_names[i] for i in selector.get_support(indices=True)]
    
    # create a DataFrame with selected features and their F-scores
    feature_scores = selector.scores_

    # access the p-values
    p_values = selector.pvalues_

    # create DataFrame
    feature_scores_df = pd.DataFrame({
        'Feature': remaining_feature_names,
        'Score': feature_scores,
        'p_value': p_values
    }).sort_values(by='Score', ascending=False)

    # get ONLY selected Features
    DF_selected = feature_scores_df[feature_scores_df.Feature.isin(selected_List)]
    
    # retrun
    return DF_selected


def getColumnName(data):
    """
    This function identifies and processes feature names from a dataset that contains a specific substring (_U). 
    It extracts and returns the base feature names before the (_U) substring.
    """
    # get features with Unknown Category
    features = data.Feature[data['Feature'].str.contains('_U')].to_list()
    
    # extract the feature name up to (but not including) '_U'
    removeFeatures = [re.search(r'^(.*?)_U', feature).group(1) if '_U' in feature else feature for feature in features]
    
    # display
    print(removeFeatures)
    
    # return
    return removeFeatures


def BayesianOptimize(model, Xdata, ydata, search_space, custom_scorer,  nJobs=-1,  nIter=64, nSplit=10, seed=RANDOM_STATE):
    """
    This function performs Bayesian Optimization to fine-tune hyperparameters for a given machine learning model. 
    It uses BayesSearchCV from scikit-optimize to search for the optimal combination of parameters over a specified 
    search space, leveraging a custom scoring metric and StratifiedKFold cross-validation.
    """
    # instantiate Stratified K-Fold cross-validation takes into account the class distribution
    kfold = StratifiedKFold(n_splits=nSplit, shuffle=True, random_state=seed)
    
    # perform Bayesian Optimization with the custom scorer
    opt = BayesSearchCV(estimator=model, search_spaces=search_space, n_iter=nIter, cv=kfold, n_jobs=-nJobs, scoring=custom_scorer)
    BayesianOpt = opt.fit(Xdata, ydata)
    
    # best model
    Bestmodel = BayesianOpt.best_estimator_
    
    # print the best parameters and best score
    print("Best parameters found: ", BayesianOpt.best_params_)
    print("Best accuracy score: ", BayesianOpt.best_score_)
    
    # display model
    print("\nCurrent Model Parameters:")
    print("\n", Bestmodel)

    return Bestmodel


# custom scoring function
def class_specific_metrics(y_true, y_pred, target_class):
    """
    Computes class-specific precision, recall, and F1 score for a given target class.
    """
    precision = metrics.precision_score(y_true, y_pred, pos_label=target_class, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, pos_label=target_class)
    f1 = metrics.f1_score(y_true, y_pred, pos_label=target_class)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# using a specific Class 1
target_class = 1
custom_scorer = metrics.make_scorer(
    lambda y_true, y_pred: class_specific_metrics(y_true, y_pred, target_class)["f1_score"], 
    greater_is_better=True
)


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