import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Regression class takes in a dataframe of values with two columns, which are respectively x and y
User can call respective functions to get regression analysis outputs
'''
class LinearRegression():
    
    def __init__(self, data) -> None:
        self.df = pd.DataFrame({'x': data.iloc[:,0], 'y': data.iloc[:,1]})
        self.beta = None
        self.alpha = None
    
    def get_alpha_beta(self):
        '''return a tuple (paried values) of beta and alpha, with beta first, alpha second'''
        x_mean = np.mean(self.df['x'])
        y_mean = np.mean(self.df['y'])
        self.df['xy_cov'] = (self.df['x'] - x_mean)* (self.df['y'] - y_mean)
        self.df['x_var'] = (self.df['x'] - x_mean)**2
        beta = self.df['xy_cov'].sum() / self.df['x_var'].sum()
        alpha = y_mean - (beta * x_mean)
        self.beta, self.alpha = beta, alpha
        
        return beta, alpha

    def predict_y(self):
        '''Obtain regression results, store into data frame, and return as an output'''
        self.get_alpha_beta()
        self.df['y_pred'] = self.alpha + self.beta*self.df['x']
        return self.df['y_pred']
    
    
    