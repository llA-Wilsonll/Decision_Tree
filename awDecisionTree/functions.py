"""
First we will create functions to carry out split search
given different forms of input and output variables.
"""

"""
calculate_logworth: Calculates the logworth given a contingency table as input

Inputs:
- contin_table: a nxm numpy arrary (usually a 2x2 matrix). 
                Columns represent observations of either side of a split for a given input variable.
                Rows represent the target values (response variable).
                The data in the table is either:
                    - the percentage of observations in each response variable category (when response in categorical)
                    - or ?avergae value of the response variable?
"""

# Using numpy as it makes creating, accessing and updating arrays much easier: pip install numpy
import numpy as np
# Using scipy so I can calculate the p-value from the chi-square statistic using the survival function (1-CDF)
from scipy import stats
# Importing log so I can calculate the logworth
from math import log

# For now this is for when the response is categorical
def calculate_logworth(contin_table):
    # Calculate Expected frequencies
    row_totals = np.sum(contin_table, axis=1)
    col_totals = np.sum(contin_table, axis=0)
    grid_total = np.sum(contin_table)
    expected = np.outer(row_totals, col_totals) / grid_total

    # Calculate Chi-squared statistic
    chi_sq = np.sum(np.divide(np.square(np.subtract(contin_table, expected)), expected))
    # Calculating the p-value from the Chi-squared statistic
    df = (contin_table.shape[0] - 1) * (contin_table.shape[1] - 1)
    p_value = stats.chi2.sf(chi_sq, df)
    # Calculate logworth from p-value
    logworth = -log(p_value, 10)
    return logworth

# print(calculate_logworth(np.array([[25,1],[1,25]])))