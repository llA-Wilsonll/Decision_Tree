"""
First we will create functions to carry out split search
given different forms of input and output variables.
"""


# Using numpy as it makes creating, accessing and updating arrays much easier: pip install numpy
import numpy as np
# Using scipy so I can calculate the p-value from the chi-square statistic using the survival function (1-CDF)
from scipy import stats
# Importing log so I can calculate the logworth
from math import log, ceil
# Using itertools so I can get all combinations of the categories of categorical variables
import itertools

"""
FUNCTION: calculate_logworth
- Calculates the logworth given a contingency table as input

Inputs:
- contin_table: a nxm numpy arrary (usually a 2x2 matrix). 
                Columns represent observations of either side of a split for a given input variable.
                Rows represent the target values (response variable).
                The data in the table is either:
                    - the number of observations in each response variable category (when response in categorical)
                    - or ?avergae value of the response variable?

NOTE: currently this is set up for CATEGORICAL response (interval response wouldn't have a contingency table populated
with frequency (number of observations) as our code is currently expecting. It would probably have mean or sum
in which case I don't believe the Chi-squared statistic would be used.
"""
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

"""
FUNCTION: list_combs
Generates a list of all possible combinations of groups given the categories in the input, not including groups 
whose complement is already in the list.

E.g. if categories = [A,B,C,D,E,F], and [A,B,C] is already in the list, then its complement [D,E,F] should not be in it.
If [A,B,C,D] is in the list, then [E,F] should not be in the list.

Inputs:
- categories: a list of the categories of which you want to take the combinations of
"""
def list_combs(categories, k):
    iterlist = list(itertools.combinations(categories, k))
    templist = iterlist  # Contains list of subsets of categories. I need to remove elements off this list, such that
                         # an element's complement in categories is not added to this list.

    for i in range(len(iterlist)):
        try:
            complement = [item for item in categories if item not in templist[i]]
        except:
            print("Iterating is complete, breaking from loop")
            break
        templist = list([item for item in templist if item != tuple(complement)])
        # update templist to remove complement before next iteration
    return templist

"""
FUNCTION: best_split
- Calculates which split for a given input variable gives highest logworth value.

For CATEGORICAL predictor variables, it will look at every combination of splitting the classes of the variable
into two groups.

For INTERVAL predictor variables then it treats every unique value is a potential split, splitting up into two groups 
of observations that are higher and observations that are lower then the given split value.

Inputs:
- Y: array of the response variable
- X: array of the predictor variable

"""
def best_split(Y, X):
    # Get unique categories of X
    categories = list(set(X))

    # Create split combinations
    groups = []
    for i in range(ceil(n / 2), n):
        groups = groups + list_combs(categories, i)

    #Loop through all combinations
       # Create contingency table
       # Calc logworth

    #Determine max logworth, split combination

    return split_groups, max_logworth
