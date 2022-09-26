import pandas as pd
import scipy.stats as stats
import numpy as np

class t_test():
    """
    A class containing methods that perform various t-tests
    
    Parameters
    ----------
    data1 : (array) array of data of interest
    data2 : (array) [optional] array of data of interest, only need to pass it for two sample test
    """
    def __init__(self, data1, data2=None) -> None:
        self.data1 = data1
        self.data2 = data2
    
    def one_sample_t_test(self, population_mean, side):
        """
        Perform one sample t-test with a side and population mean
        
        Parameters
        ----------
        population_mean : (float) population mean to be tested
        side : (str) only allows 'two-sided', 'less', 'greater', side of the test to perform
        
        Returns
        -------
        t-statistic (float)
        """
        if side not in ['two-sided', 'less', 'greater']:
            raise Exception("Only accept 'two-sided', 'less', or 'greater' for parameter 'side'")
        return stats.ttest_1samp(self.data1, population_mean, alternative=side)
    
    def two_sample_t_test(self, side):
        """
        Perform two sample t-test between data1 and data2
        
        Parameters
        ----------
        side : (str) only allows 'two-sided', 'less', 'greater', side of the test to perform
        
        Returns
        -------
        t-statistic (float)
        """
        if side not in ['two-sided', 'less', 'greater']:
            raise Exception("Only accept 'two-sided', 'less', or 'greater' as a parameter")
        return stats.ttest_ind(self.data1, self.data2, alternative=side)
    
    def paired_sample_t_test(self):
        """Perform paired sample t-test between data1 and data2
        
        Returns
        -------
        t-statistic (float)
        """
        return stats.ttest_rel(self.data1, self.data2)