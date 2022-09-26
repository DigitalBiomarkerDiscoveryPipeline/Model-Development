import pandas as pd
import numpy as np

class LinearRegression():
    """
    Regression class takes in a dataframe of values with two columns, which are respectively x and y
    User can call respective functions to get regression analysis outputs
    
    Parameters
    ----------
    df : (pandas.DataFrame) a pandas dataframe containing two columns, first being x-values, second
    being y-values
    """
    
    def __init__(self, data) -> None:
        self.df = pd.DataFrame({'x': data.iloc[:,0], 'y': data.iloc[:,1]})
        self.beta = None
        self.alpha = None
    
    def get_alpha_beta(self):
        """
        Function that gets alpha and beta of the data in DataFrame
        
        Returns
        -------
        a tuple (paried values) of beta and alpha, with beta first, alpha second"""
        x_mean = np.mean(self.df['x'])
        y_mean = np.mean(self.df['y'])
        self.df['xy_cov'] = (self.df['x'] - x_mean)* (self.df['y'] - y_mean)
        self.df['x_var'] = (self.df['x'] - x_mean)**2
        beta = self.df['xy_cov'].sum() / self.df['x_var'].sum()
        alpha = y_mean - (beta * x_mean)
        self.beta, self.alpha = beta, alpha
        
        return beta, alpha

    def predict_y(self):
        """
        Obtain regression results, store into data frame, and return as an output
        
        Returns
        -------
        A column of DataFrame of predicted y-values
        """
        self.get_alpha_beta()
        self.df['y_pred'] = self.alpha + self.beta*self.df['x']
        return self.df['y_pred']

from sklearn.svm import SVR
def run_svr(data_in, x_data, y_data, kernel='rbf', degree=3, gamma='scale', tol=1e-3, c=1.0, epsilon=0.1, cache_size=200, verbose=False):
    """
    run support vector regression using library from scikit learn

    Parameters
    ----------
    data_in : array or float
        data to be analyzed and predicted based on model
    x_data : array
        x values of data
    y_data : array
        y values of data
    kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} , optional
       Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will be used. 
       If a callable is given it is used to precompute the kernel matrix., by default 'rbf'
    degree : int, optional
        Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels., by default 3
    gamma : {‘scale’, ‘auto’} or float, optional
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’., by default 'scale'
    tol : float, optional
        tolerance for stopping criterion, by default 1e-3
    c : float, optional
        Regularization parameter. The strength of the regularization is inversely proportional to C. 
        Must be strictly positive. The penalty is a squared l2 penalty., by default 1.0
    epsilon : float, optional
        Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in 
        the training loss function with points predicted within a distance epsilon from the actual value., by default 0.1
    cache_size : int, optional
        Specify the size of the kernel cache (in MB)., by default 200
    verbose : bool, optional
        Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm 
        that, if enabled, may not work properly in a multithreaded context., by default False

    Returns
    -------
    array or float
        predicted values from data_in
    """
    svr = SVR(kernel, degree, gamma, tol, c, epsilon, cache_size, verbose)
    svr.fit(x_data, y_data)
    y_pred = svr.predict(data_in)
    return y_pred

from sklearn.tree import DecisionTreeRegressor
def run_decision_tree(data_in, x_data, y_data, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Run regression with decision tree from scikit learn

    Parameters
    ----------
    data_in : array or float
        data to be predicted from fitted model
    x_data : array
        x values for the regression
    y_data : array
        y values for the regression
    criterion : {“squared_error”, “friedman_mse”, “absolute_error”, “poisson”}, optional
        The function to measure the quality of a split. 
        Supported criteria are “squared_error” for the mean squared error, which is equal to variance reduction as 
        feature selection criterion and minimizes the L2 loss using the mean of each terminal node, “friedman_mse”, 
        which uses mean squared error with Friedman’s improvement score for potential splits, “absolute_error” for 
        the mean absolute error, which minimizes the L1 loss using the median of each terminal node, and “poisson” 
        which uses reduction in Poisson deviance to find splits., by default 'squared_error'
        
    splitter : {“best”, “random”}, optional
       The strategy used to choose the split at each node. 
       Supported strategies are “best” to choose the best split and “random” to choose the best random split., by default 'best'
       
    max_depth : int, optional
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples., by default None
        
    min_samples_split : int or float, optional
        The minimum number of samples required to split an internal node:

        If int, then consider min_samples_split as the minimum number.
        If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split., by default 2
        
    min_samples_leaf : int or float, optional
        The minimum number of samples required to be at a leaf node. 
        A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples 
        in each of the left and right branches. This may have the effect of smoothing the model, especially in regression., by default 1

    Returns
    -------
    array or float
        predicted values from data_in
    """
    regressor = DecisionTreeRegressor(criterion, splitter, max_depth, min_samples_split, min_samples_leaf)
    regressor.fit(x_data, y_data)
    y_predict = regressor.predict(data_in)
    return y_predict

from sklearn.ensemble import RandomForestRegressor
def run_random_foreset(data_in, x_data, y_data, n_estimators=100, criterion='squared error', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0):
    """
    run random forest regression with fitted data and data_in

    Parameters
    ----------
    data_in : array or float
        data to be predicted from the learned models
    x_data : array
        array of x values of data to be fitted
    y_data : array
        array of y values of data to be fitted
    n_estimators : int, optional
        number of trees in the forest, by default 100
    criterion : {“squared_error”, “absolute_error”, “poisson”}, optional
        The function to measure the quality of a split. Supported criteria are “squared_error” for the mean squared error, 
        which is equal to variance reduction as feature selection criterion, “absolute_error” for the mean absolute error, 
        and “poisson” which uses reduction in Poisson deviance to find splits. 

        Training using “absolute_error” is significantly slower than when using “squared_error”., by default 'squared error'
        
    max_depth : int, optional
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples., by default None
        
    min_samples_split : int or float, optional
        The minimum number of samples required to split an internal node:

            If int, then consider min_samples_split as the minimum number.
            If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split., by default 2
            
    min_samples_leaf : int or float, optional
        The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression., by default 1
        
    max_features : {“sqrt”, “log2”, None} int or float, optional
        The number of features to consider when looking for the best split:

            If int, then consider max_features features at each split.
            If float, then max_features is a fraction and round(max_features * n_features) features are considered at each split.
            If “auto”, then max_features=n_features.
            If “sqrt”, then max_features=sqrt(n_features).
            If “log2”, then max_features=log2(n_features).
            If None or 1.0, then max_features=n_features.
        
        , by default 1.0

    Returns
    -------
    array or float
        predicted data from random forest regressor using data_in passed by user
    """
    regressor = RandomForestRegressor(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features)
    regressor.fit(x_data, y_data)
    y_predict = regressor.predict(data_in)
    return y_predict

import xgboost as xgb
def run_xgboost(data_in, x_data, y_data, n_estimators, max_depth, max_leaves, max_bin, grow_policy, learning_rate, verbosity, gamma):
    """
    Run xgboost regression fitted with x_data and y_data, and predict using data_in

    Parameters
    ----------
    data_in : array or float
        data to be predicted from regression
    x_data : array
        x values of data for regression
    y_data : array
        y values of data for regression
    n_estimators : int
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : int
        maximum tree depth
    max_leaves : int
        Maximum number of leaves; 0 indicates no limit.
    max_bin : int
        If using histogram-based algorithm, maximum number of bins per feature
    grow_policy : 0 or 1
        Tree growing policy. 
        0: favor splitting at nodes closest to the node, i.e. grow depth-wise. 
        1: favor splitting at nodes with highest loss change.
    learning_rate : float
        boosting learning rate
    verbosity : int
        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
    gamma : float
         Minimum loss reduction required to make a further partition on a leaf node of the tree.

    Returns
    -------
    array or float
        predicted values from data_in after regression
    """
    regressor = xgb.XGBRegressor(n_estimators, max_depth, max_leaves, max_bin, grow_policy, learning_rate, verbosity, gamma=gamma)
    regressor.fit(x_data, y_data)
    pred = regressor.predict(data_in)
    return pred