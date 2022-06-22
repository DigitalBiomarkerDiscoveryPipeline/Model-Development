import pandas as pd
import scipy.stats as stats
import numpy as np

'''
GUIDELINE: pass data as an array(s) into T-test class
Then use functions in this class to get desired results
'''

class t_test():
    
    def __init__(self, data1, data2=None) -> None:
        self.data1 = data1
        self.data2 = data2
    
    def one_sample_t_test(self, population_mean, side):
        if side not in ['two-sided', 'less', 'greater']:
            raise Exception("Only accept 'two-sided', 'less', or 'greater' for parameter 'side'")
        return stats.ttest_1samp(self.data1, population_mean, alternative=side)
    
    def two_sample_t_test(self, side):
        if side not in ['two-sided', 'less', 'greater']:
            raise Exception("Only accept 'two-sided', 'less', or 'greater' as a parameter")
        return stats.ttest_ind(self.data1, self.data2, alternative=side)
    
    def paired_sample_t_test(self):
        return stats.ttest_rel(self.data1, self.data2)
    
    
    
        
    