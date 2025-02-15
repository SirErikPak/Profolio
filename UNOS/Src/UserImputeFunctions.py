from sklearn.mixture import GaussianMixture
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def findGaussianMixtureComponents(data, seed, component_range=range(1, 10), figsize=(8,6)):
    """
    Determines the optimal number of components for a Gaussian Mixture Model (GMM)
    using Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Parameters:
    ----------
    data : pandas.DataFrame or numpy.ndarray
        The dataset to fit the GMM. It should be a 2D array-like structure with samples as rows and features as columns.
    seed : int
        Random seed for reproducibility.
    component_range : range or list, optional, default=range(1, 10)
        Range of component numbers to evaluate.
    figsize : tuple, optional, default=(8, 6)
        Size of the figure for plotting.

    Returns:
    -------
    None

    Notes:
    -----
    The function fits GMMs with different numbers of components to the provided data,
    computes their AIC and BIC scores, and plots these scores to help visualize the optimal
    number of components. The optimal number is determined based on the lowest BIC score,
    as BIC tends to favor models that generalize well to unseen data by penalizing complexity
    more strongly than AIC.
    """

    # Ensure data is in numpy array format
    if hasattr(data, 'values'):
        data = data.values

    # Remove NaNs
    data = data[~np.isnan(data).any(axis=1)]

    # Initialize lists to store AIC and BIC scores
    aic_scores = []
    bic_scores = []

    for n_components in component_range:
        # Initialize the GMM
        gmm = GaussianMixture(n_components=n_components, random_state=seed)

        # Fit the GMM to the data
        gmm.fit(data)

        # Compute AIC and BIC
        aic_scores.append(gmm.aic(data))
        bic_scores.append(gmm.bic(data))

    # Plot AIC and BIC scores
    plt.figure(figsize=figsize)
    plt.plot(component_range, aic_scores, label='AIC', marker='o')
    plt.plot(component_range, bic_scores, label='BIC', marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.title('AIC and BIC Scores for Different Number of Components')
    plt.legend()
    plt.show()

    # Select the optimal number of components based on the lowest BIC
    optimal_n_components = component_range[np.argmin(bic_scores)]
    print(f'Optimal number of components: {optimal_n_components}')


def imputeGaussianMixture(dataSeries, seed, n_components, minimum, maximum):
    """
    Imputes missing values in a data series using a Gaussian Mixture Model (GMM).

    Parameters:
    ----------
    dataSeries : pandas.Series or array-like
        The data series containing missing values to be imputed.
    seed : int
        Random seed for reproducibility.
    n_components : int
        The number of mixture components to use in the GMM.
    minimum : float
        The minimum value to clip the imputed values.
    maximum : float
        The maximum value to clip the imputed values.

    Returns:
    -------
    imputed_data : pandas.Series
        The data series with missing values imputed.
    """
    # Convert series to numpy array
    data = np.asarray(dataSeries)

    # Fit GMM to the observed data
    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    observed_data = data[~np.isnan(data)].reshape(-1, 1)
    gmm.fit(observed_data)

    # Impute missing values by sampling from the GMM
    missing_mask = np.isnan(data)
    imputed_values, _ = gmm.sample(np.sum(missing_mask))

    # Clip the imputed values to stay within the range
    imputed_values = np.clip(imputed_values, minimum, maximum)

    # Fill missing data with the imputed values
    data[missing_mask] = imputed_values.flatten()

    # Convert the numpy array back to a pandas DataFrame
    imputed_data = pd.DataFrame(data, index=dataSeries.index, columns=dataSeries.columns)

    return imputed_data


def imputeKNN(data, featuresList, n_neighbors=range(1, 10), cv=5, figsize=(8,6), flag=True):
    """
    Impute missing values using KNN and find the optimal number of neighbors and weights.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the features to impute.
    features_list (list): List of feature names to impute.
    n_neighbors (range): Range of n_neighbors to try. Default is range(1, 10).
    cv (int): Number of cross-validation folds. Default is 5.
    figsize (tuple): Figure size for the plot. Default is (8, 6).

    Returns:
    pd.DataFrame: DataFrame with imputed values for the specified features.
    """
    # Select features you want to impute
    features_to_impute = data[featuresList]
    
    # Define the parameter grid for n_neighbors and weights
    if flag:
        param_grid = {'n_neighbors': n_neighbors, 'weights': ['uniform', 'distance']}
    else:
        param_grid = {'n_neighbors': n_neighbors, 'weights': ['uniform']}

    # Use GridSearchCV to find the best n_neighbors and weights
    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=cv)
    grid_search.fit(features_to_impute.dropna(), features_to_impute.dropna())

    # Initialize variable & print
    best_params = grid_search.best_params_
    best_n_neighbors = best_params['n_neighbors']
    best_weights = best_params['weights']
    print(f"The best n_neighbors for KNN imputer is: {best_n_neighbors}")
    print(f"The best weights for KNN imputer is: {best_weights}")

    # Plot the results of GridSearchCV
    results = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=figsize)
    if flag:
        for weight in ['uniform', 'distance']:
            subset = results[results['param_weights'] == weight]
            plt.plot(n_neighbors, subset['mean_test_score'], label=f'weights={weight}')
    else:
        for weight in ['uniform']:
            subset = results[results['param_weights'] == weight]
            plt.plot(n_neighbors, subset['mean_test_score'], label=f'weights={weight}')        
    
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Mean Test Score')
    plt.title('Grid Search Results for KNN Imputer')
    plt.legend()
    plt.show()

    # Impute using the best n_neighbors and weights
    imputer = KNNImputer(n_neighbors=best_n_neighbors, weights=best_weights)
    imputed_data = imputer.fit_transform(features_to_impute)

    # Return imputed DataFrame
    return pd.DataFrame(imputed_data, columns=features_to_impute.columns)