import pandas as pd
import numpy as np


def getFeatureList(data, string):
    # initialize features and sort them alphabetically
    features = sorted(data.columns[data.columns.str.contains(string)].tolist())

    # display
    print(data[features].describe(include='all').T.to_string())
    print("\n:::: NaN Count:")
    print(data[features].isna().sum().sort_index().to_string())
    return features


def writeToFile(data, filename, path='../Data/', format='csv'):
    """
    write dataframe to disk
    """
    # intialize variable
    file_path = path + filename + f".{format}"

    if format.lower() == 'csv':
        # write to disk
        data.to_csv(file_path, index=False)
    else:
        data.to_pickle(file_path)     
    
    return print(f"{len(data):,} records written to {file_path}")



def mappingCol(data, colstr, mapdict, display=True):
    """
    This function is designed to map values in a specified column of a DataFrame to new values 
    using a provided dictionary and optionally print the unique values after the mapping and converts 
    the column to the category data type.
    """
    # map values in the column using the mapping dictionary, leaving values unchanged if not found in the mapping
    data[colstr] = data[colstr].map(mapdict).fillna(data[colstr])
    # convert datatype to categorical data type
    data[colstr] = data[colstr].astype('category')

    if display:
        print(f"Converted Column {colstr} Unique Vaue(s) {data[colstr].unique()}")

    return data



def toCategory(data, col):
    """
    This function converts specified columns in a DataFrame to the category data type.
    """
    # iterate
    for val in col:
        data[val] = data[val].astype('category') 

    return data



def percentageNull(data):
    """
    This function calculates the percentage of missing values (NaN) for each column in a DataFrame 
    and returns those columns that contain any missing values, along with their corresponding NaN 
    count and percentage.
    """
    # calculate the percentage of non-null values for each column
    per_calc = pd.DataFrame(100 - (data.count() / len(data) * 100))
    
    # rename columns name
    per_calc = per_calc.rename(columns={0: 'percentage'})

    # add counter
    per_calc['NaNCount'] = data.isna().sum()
    
    # sort
    per_calc = per_calc.sort_values(by='percentage', ascending=False)

    # get count not zero
    NanReturn = per_calc[per_calc.NaNCount != 0]
    
    return NanReturn



def removeRowUsingMask(data, removeColLst, colstr, string='', display=True):
    # initialized variable
    pRow = data.shape[0]
    # boolean mask
    mask = ~data[colstr].isin(removeColLst)
    
    # apply the mask to keep only rows where 'removeColLst'
    data = data[mask]
    
    # reset the index if needed
    data = data.reset_index(drop=True)

    # initialized variable
    cRow = data.shape[0]
    
    if display:
        if string == '':
            # disply row removed msg
            print(f"Remove {pRow - cRow} row(s) from {colstr} column in a DataFrame.")
        else:
            print(f"Remove {pRow - cRow} row(s) from {string} DataFrame.")            

    return data

    

def removeColumn(data, col, display=True):
    """
    Remove unwanted columns if they exist in the DataFrame
    """
    # Ensure col_list is a list
    col_list = col.tolist() if isinstance(col, np.ndarray) else col
    columns_set = set(data.columns)
    # # Ensure col is a list of column names (strings)
    # if isinstance(col, pd.Index):
    #     col_list = col.tolist()  # Convert Index to list
    # else:
    #     col_list = col if isinstance(col, list) else [col]
    
    # # Convert columns to a set for efficient lookup
    # columns_set = set(data.columns)
    
    existing_cols = [c for c in col_list if c in columns_set]
    non_existing_cols = [c for c in col_list if c not in columns_set]
    
    if display:
        if existing_cols:
            print(f"\nRemoved Features: {sorted(existing_cols)}")
        if non_existing_cols:
            print(f"\nSkipped non-existing columns: {sorted(non_existing_cols)}")
        print(f"\nTotal Row(s) & Column(s) Before Removing Column(s): {data.shape[0]:,} & columns: {data.shape[1]:,}")
    
    if existing_cols:
        data = data.drop(columns=existing_cols, axis=1)
        data = data.reset_index(drop=True)
    
    if display:
        print(f"Total Row(s) & Column(s) After Removing Column(s): {data.shape[0]:,} & columns: {data.shape[1]:,}")

    return data


def maintainDataDict(data, removeColLst, txt= '', col="Information", display=True):
    """
    Maintain data dictionary
    """
    # get indices
    idx = data[data.Feature.isin(removeColLst)].index
    # append to exiting info
    data.loc[idx,col] = f"{txt} - " + data[col]

    if display:
        # disply update msg
        print(f"Data Dictionary Updated.")

    return data


def HouseKeeping(data, removeColLst, dataDict,  dataLabel, dataCAN, dataDON, dataBoth, dataOrdinal, dataNominal, dataNumeric, dataDrop, dataObject, dataUnknown, dataDate, txt, display=True):
    """
    Run helper fuction for house keeping
    """
    # update data dictionary (house keeping)
    dataDict = maintainDataDict(dataDict, removeColLst, txt, display=display)
    
    # remove DataFrame data (house keeping)
    dataLabel = removeRowUsingMask(dataLabel, removeColLst, colstr='column', string='df_label', display=display)
    dataCAN = removeRowUsingMask(dataCAN, removeColLst, colstr='column', string='df_can', display=display)
    dataDON = removeRowUsingMask(dataDON, removeColLst, colstr='column', string='df_don', display=display)
    dataBoth = removeRowUsingMask(dataBoth, removeColLst, colstr='column',  string='df_both', display=display)
    dataOrdinal = removeRowUsingMask(dataOrdinal, removeColLst, colstr='column',  string='df_ordinal', display=display)
    dataNominal = removeRowUsingMask(dataNominal, removeColLst, colstr='column', string='df_nominal',  display=display)
    dataNumeric = removeRowUsingMask(dataNumeric, removeColLst, colstr='column', string='df_numeric',  display=display)
    dataDrop = removeRowUsingMask(dataDrop, removeColLst, colstr='column',  string='df_drop', display=display)
    dataObject = removeRowUsingMask(dataObject, removeColLst, colstr='column',  string='df_object', display=display)
    dataUnknown = removeRowUsingMask(dataUnknown, removeColLst, colstr='column',  string='df_unknown', display=display)
    dataDate = removeRowUsingMask(dataDate, removeColLst, colstr='column',  string='df_date', display=display)
    
    # remove features
    data = removeColumn(data, removeColLst, display=display)


    return data, dataDict, dataLabel, dataCAN, dataDON, dataBoth, dataOrdinal, dataNominal, dataNumeric, dataDrop, dataObject, dataUnknown, dataDate


def dataDictSearch(datadic, colList, indexView=True):
    """
    This function is designed to search for a given column name (col) in a DataFrame (datadic) 
    based on the featureName column. 
    """
    # dataframe to display
    data = datadic.loc[:,['Feature','Description', 'FormSection', 'DataType', 'SASAnalysisFormat', 'Comment', \
                          'Information']][datadic.Feature.isin(colList)]
    # display
    print(data.to_string(index=indexView))
    



def featureInfo(data, datadict, strCol, unique=False, indexView=True):
    """
    Return column(s) and display descriptive statistics/NaN count/Datatypes, 
    including displaying unique records.
    """
    # feature name & index
    feature = data.columns[data.columns.str.contains(strCol)].to_list()
    idx = datadict[datadict.Feature.isin(feature)].index
    if len(feature) == 0:
        raise ValueError(f"No matching features found for '{strCol}'")
    else:
        # descriptive statistics
        print(data[feature].describe(include='all').T.to_string(index=indexView))
        
        # check NaN & datatype
        print('\nNaNs:')
        print(data[feature].isna().sum())
        print('\nDatatypes:')
        print(data[feature].dtypes)
        print('\n')
        # dataframe to display
        print(datadict.loc[:,['Feature','Description', 'FormSection', 'DataType', 'SASAnalysisFormat', 'Comment', 'Information']]\
              [datadict.Feature.isin(feature)].to_string(index=indexView))
        print('\n')

        # iterate over feature if flag TRUE
        if unique:
            for i in feature:
                print(f"{i}: {data[i].unique()}")

        return feature, idx
    


def insertIntoDataFrame(data, values):
    """
    This function inserts each value from 
    the list values as a new row in the 
    DataFrame, and avoiding duplicates.
    """
    for value in values:
        # Convert value to a Series for easier comparison
        new_row = pd.Series(value, index=data.columns)
        
        # Check if this row already exists in the DataFrame
        if not (data == new_row).all(1).any():
            # doesn't exist, add it
            data.loc[len(data)] = value
        else:
            print(f"Already Exists: ({value})")
    
    return data



def mappingDataAndDictionary(data, dataDict, mapping, indices, Type='', txt=''):
    """
    The function mappingDataAndDictionary is designed to map column names of a dataset (data) and 
    update its associated data dictionary (dataDict) based on the mapping provided.
    """        
    # rename data
    data = data.rename(columns=mapping)
    # update dictionary
    dataDict['Feature'] = dataDict['Feature'].map(mapping).fillna(dataDict['Feature'])
    dataDict.loc[indices, 'Information'] = f"{txt}"
    dataDict.loc[indices, 'FeatureType'] = Type

    return data, dataDict



def updateDictionaryInformation(dataDict, indices, txt='', FeatureType=None):
     # update dictionary
    dataDict.loc[indices, 'Information'] = f"{txt}"
    if FeatureType:
        dataDict.loc[indices, 'FeatureType'] = f"{FeatureType}"
    return dataDict



def symmetricDifference(set1, set2):
    """
    This function calculates the symmetric difference between two sets, set1 and set2. 
    """
    # calculate symmetric difference
    symmetric_difference = set1.symmetric_difference(set2)
    # sort
    symmetric_difference = sorted(symmetric_difference)
    print(f"Symmetric difference: {symmetric_difference}")



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



def categoryContingency(data, group, label, observed=False, fill_value=0):
    """
    This function creates a contingency table for analyzing the frequency distribution 
    of a categorical label across different groups. It also calculates row-wise percentages
    for 'Dead' and 'Living' per category, including totals for the last row.
    """
    # Create contingency table
    data = data.groupby(group, observed=observed)[label].value_counts().unstack(fill_value=fill_value)
    
    # Get index name
    indexName = data.index.name
    
    # Add row totals
    data['Row Total'] = data.sum(axis=1)
    
    # Calculate row-wise percentages
    if 'Dead' in data.columns:
        data['Dead %'] = (data['Dead'] / data['Row Total']) * 100
    else:
        data['Dead %'] = 0

    if 'Living' in data.columns:
        data['Living %'] = (data['Living'] / data['Row Total']) * 100
    else:
        data['Living %'] = 0
    
    # Add column totals
    totals = data.sum().to_frame().T
    totals.index = ['Column Total']
    
    # Ensure percentages and totals for 'Living' and 'Dead' are correct
    totals['Dead %'] = (totals['Dead'] / totals['Row Total']) * 100 if 'Dead' in totals.columns else 0
    totals['Living %'] = (totals['Living'] / totals['Row Total']) * 100 if 'Living' in totals.columns else 0
    
    # Concatenate the totals row to the original DataFrame
    df_with_totals = pd.concat([data, totals])
    
    # Set index name
    df_with_totals.index.name = indexName
    
    return df_with_totals



def categoryContingencySurvival(data, group, observed=False, fill_value=0):
    """
    This function creates a contingency table for analyzing the frequency distribution 
    of a categorical label across different groups. It also calculates row-wise percentages
    for 'Dead' and 'Living' per category, including totals for the last row.
    """
    # initialize variable
    label = 'Survival'
    # replace True/False with Living/Dead
    data[label] = data[label].replace({True: 'Living', False: 'Dead'})

    # Create contingency table
    data = data.groupby(group, observed=observed)[label].value_counts().unstack(fill_value=fill_value)
    
    # Get index name
    indexName = data.index.name
    
    # Add row totals
    data['Row Total'] = data.sum(axis=1)
    
    # Calculate row-wise percentages
    if 'Dead' in data.columns:
        data['Dead %'] = (data['Dead'] / data['Row Total']) * 100
    else:
        data['Dead %'] = 0

    if 'Living' in data.columns:
        data['Living %'] = (data['Living'] / data['Row Total']) * 100
    else:
        data['Living %'] = 0
    
    # Add column totals
    totals = data.sum().to_frame().T
    totals.index = ['Column Total']
    
    # Ensure percentages and totals for 'Living' and 'Dead' are correct
    totals['Dead %'] = (totals['Dead'] / totals['Row Total']) * 100 if 'Dead' in totals.columns else 0
    totals['Living %'] = (totals['Living'] / totals['Row Total']) * 100 if 'Living' in totals.columns else 0
    
    # Concatenate the totals row to the original DataFrame
    df_with_totals = pd.concat([data, totals])

    return df_with_totals



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
    df_str = datadic.loc[:,['Feature','Description', 'DataType', 'SASAnalysisFormat', 'Comment', 'FeatureType', 'Information']][datadic.Feature.str.contains(parm) & \
                ~datadic.Information.str.contains('DROP', case=False, na=False)]

    if flag:
        feature = datadic.Feature[datadic.Feature.str.contains(parm) & ~datadic.Information.str.contains('DROP', case=False, na=False)].tolist()
        return feature
    else:
        return df_str