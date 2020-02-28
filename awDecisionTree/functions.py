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
# Using pandas as input data will likely be imported as a pandas DataFrame
import pandas as pd

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
            max_split = split
            max_logworth = logworth
            max_contin = contin

    return max_split, max_logworth, max_contin


"""
FUNCTION: final_split_point
- Calculates which variable has the best split given an input data set (Pandas dataframe)

It compares the logworth for the best split for each predictor variable and determines which split overall is the best

Inputs:
- df_input: Pandas dataframe of the input dataset (including predictor and response variables)
- y_index: index of the column in df_input that is the response variable (Y)
"""


def final_split_point(df_input, y_index):
    y_values = list(df_input.iloc[:, y_index])  # extracting y values from Pandas dataframe into a list

    final_logworth = 0.0  # initialising logworth to 0
    for i in [x for x in range(len(df_input.columns)) if x != y_index]:
        x_values = list(df_input.iloc[:, i])  # extracting x values from Pandas dataframe into a list
        x_name = df_input.columns[i]  # name of the current X variable being used
        max_split, max_logworth, max_contin = best_split(Y=y_values, X=x_values)

        # Calculating best split so far at each iteration
        # Note: this will always be entered on first iteration as logworth is always >0
        if max_logworth > final_logworth:
            final_split = max_split
            split_variable = x_name
            final_logworth = max_logworth
            final_contin = max_contin

    return final_split, split_variable, final_logworth, final_contin


if __name__ == "__main__":
    df_in = df = pd.DataFrame([['A', 'B', 0, 'C'],
                               ['Z', 'b', 0, 'another'],
                               ['W', 'B', 1, 'two'],
                               ['A', 'BB', 0, 'another'],
                               ['A', 'BB', 1, 'another'],
                               ['Z', 'BB', 0, 'C'],
                               ['W', 'B', 0, 'C'],
                               ['A', 'b', 1, 'two'],
                               ['Z', 'B', 1, 'two'],
                               ['A', 'b', 1, 'C'],
                               ['A', 'B', 0, 'C'],
                               ['W', 'b', 1, 'another'],
                               ['A', 'b', 1, 'C'],
                               ['A', 'BB', 1, 'two'],
                               ['W', 'BB', 0, 'C'],
                               ['A', 'b', 0, 'another'],
                               ['W', 'B', 0, 'C']], columns=['x1', 'x2', 'y', 'x3'])
    split, split_variable, logworth, contin = final_split_point(df_in, 2)
    # Determine max logworth, split combination
    print("Split Variable: {}".format(split_variable))
    print("Best Split: {}".format(split))
    print("Max logworth: {}".format(logworth))
    print("Contingency Table:\n{}".format(contin))
