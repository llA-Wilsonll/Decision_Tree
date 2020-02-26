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
    categories = list(set(X))  # Get unique categories of X
    categories.sort()  # Sorting so complement doesn't have elements in wrong order
    n = len(categories)

    # Create split combinations
    split_groups = []
    for i in range(ceil(n / 2), n):
        # A list of combinations of values in the list "categories" that are of size i
        test_list = [subset for subset in itertools.combinations(categories, i)]

        # If n is even, then I need to remove elements off this list, such that an element's complement in categories
        # is not added to this list.
        if i == (n / 2):
            for subset in test_list:
                complement = np.sort(np.setdiff1d(categories, subset))  # Adding sort to ensure correct order before
                # using remove method
                complement = tuple(complement)  # Making a tuple as that is the form of the combinations in test_list
                test_list.remove(complement)  # Removing the complement

        split_groups = split_groups + test_list

    # Loop through all combinations and determine which one has the maximum logworth
    max_logworth = 0.0

    for split in split_groups:
        contin = np.zeros([len(set(Y)), 2])  # Number of rows is equal to the number of unique categories of Y

        # Create contingency table
        for k in range(len(X)):
            if X[k] in split:
                contin[Y[k], 0] += 1  # Y needs to have its categories encoded as 0,1,2,.....
            else:
                contin[Y[k], 1] += 1

        # Calc logworth
        logworth = calculate_logworth(contin)

        # Calculating best split so far at each iteration
        # Note: this will always be entered on first iteration as logworth is always >0
        if logworth > max_logworth:
            max_logworth = logworth
            max_split = split
            max_contin = contin

    return max_contin, max_logworth, max_split


if __name__ == "__main__":
    max_contin, max_logworth, max_split = best_split(Y=[0, 1, 0, 0, 1], X=[2, 5, 2, 10, 4])
    # Determine max logworth, split combination
    print("Contingency Table:\n{}".format(max_contin))
    print("Max logworth: {}".format(max_logworth))
    print("Best Split: {}".format(max_split))
