# Stats Models
## T-Test Tutorial
1. User get array (or numpy array) of data from pre-processed module, then pass into t_test() class, t_test() can have either 1 data or 2 data. For instance, t_test(data1) and t_test(data1, data2) both works depending on whether user want to test one sample or two samples
2. Call functions on t_test() class to get desired values

```python
# For one sample t-test, call below function to get t-test statistic based on that user wants to test
t_test(data1).one_sample_t_test(mean, 'two-sided')      # For two-sided test
t_test(data1).one_sample_t_test(mean, 'less')           # For one-sided, less than
t_test(data1).one_sample_t_test(mean, 'greater')        # For one-sided, greater than 

# For two sample t-test, call below function to get t-test statistic based on side of the test
t_test(data1, data2).two_sample_t_test('two-sided')     # For two-sided test
t_test(data1, data2).two_sample_t_test('less')          # For one-sided, less than
t_test(data1, data2).two_sample_t_test('greater')       # For one-sided, greater than

# For paired sample t-test, simply call below function to get t-test statistic
t_test(data1, data2).paired_sample_t_test()
```

# ML Models
# DL Models
