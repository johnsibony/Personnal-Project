"""
List of useful functions for Term structure strategies.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

import numpy as np
import pandas as pd
from datetime import timedelta

def term_structure(price, threshold=0.7):
    """Function that return a score in [-1,1] of the term structure : contango (positive), backwardation (negative).
    The score is computed according to the correlation coefficient between the contract prices and the prices of a perfect contango structure, adjusting with the 'threshold' argument.
    Correlation above 'threshold' wwould be renormalized in [0,1]. Correlation below 'threshold' would be renormalized in [-1,1].
    :param price: Prices of the contract with increasing maturity.
    :param threshold: Correlation coefficient of the contango/backwardation frontiere. If correlation>'threshold', contango structure. Otherwise, backwardation structure."""    
    def perfect_contango(low, high):
        """The perfect contango structure is computed by creating a concave curve between the minimum and maximum contract prices.
        :param low: Minimum price of 'price'.
        :param high: Maximum price of 'price'."""
        amplitude = high - low
        coefficient = np.array([0, 0.3, 0.55, 0.7, 0.8 ,0.88, 0.94, 0.97, 1])
        return low + coefficient * amplitude
    n = min(len(price), 9)
    if(n==1):
        return
    else:
        price = price[0:n]
        low, high = min(price), max(price)
        contango = perfect_contango(low, high)
        keep_index = [round(i*8/(n-1),0) for i in range(n)] #take n indexes equally separated among the 9 indexes of perfect contango.
        contango = [contango[int(ind)] for ind in keep_index]
        corr = np.corrcoef(price, contango)[0,1]
        sign = np.sign(int(corr>threshold)-0.5)
        score = (corr-threshold) / (1-sign*threshold)
        return score

def front_end_month(date):
    """Returns the last date of the month of the date 'date'. If the date 'date' is already a end of month date, returns the last date of the following month.
    :param date: Date in datetime format."""
    eoms = pd.date_range(start=date, end=date+timedelta(days=50), freq='BM').date
    if(date==eoms[0]):
        eom = eoms[1]
    else:
        eom = eoms[0]
    return eom

if __name__ == '__main__':
    term_structure([20,50,75,90,98,105,108])
    front_end_month(str_to_datetime('2006-01-31'))
    front_end_month(str_to_datetime('2006-01-02'))