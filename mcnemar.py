#!/usr/bin/env python
# -*- coding: utf-8 -*-
# McNemar's test source: https://machinelearningmastery.com/mcnemars-test-for-machine-learning/

import statsmodels.stats.contingency_tables as sm

"""
@input - truth values of the predictions of the first model (List); truth values of the predictions of the second model (List)
         
Method that calculates the contignecy table and evaluates the statistical significance of the predictions made
by two classifiers using the McNemar's test. Prints out the contigency table, test statistic and p-value of the test.
Prints out if the null hypothesis was rejected or not with the alpha value of 0.05.

@output - Test statistic, p-value and hypothesis fail/reject result as string
"""


def mcnemar_test(p1, p2):
    con_table = [[0, 0], [0, 0]]
    for i, p1_val in enumerate(p1):
        p2_val = p2[i]
        if p1_val == True and p2_val == True:
            con_table[0][0] += 1
        elif p1_val == True and p2_val == False:
            con_table[0][1] += 1
        elif p1_val == False and p2_val == True:
            con_table[1][0] += 1
        elif p1_val == False and p2_val == False:
            con_table[1][1] += 1
    print("Contingency table: {}".format(con_table))

    # test statistic must be calculated using binomial distribution if any of the table values are less than 25
    if any(val < 25 for entry in con_table for val in entry):
        print("Some value < 25. Calculating exact p-value")
        result = sm.mcnemar(con_table, exact=True)
    else:
        print("All values >= 25. Calculating standard McNemar's statistic")
        result = sm.mcnemar(con_table, exact=False, correction=True)

    print('statistic=%.3f, p-value=%.5f' % (result.statistic, result.pvalue))
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
        return result.statistic, result.pvalue, 'fail'
    else:
        print('Different proportions of errors (reject H0)')
        return result.statistic, result.pvalue, 'reject'
