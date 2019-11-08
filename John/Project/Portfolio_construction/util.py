"""
List of useful function for backtesting
"""

# Author: John Sibony <john.sibony@hotmail.fr>

from itertools import chain
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

def str_to_datetime(date):
    """Convert string date to datetime format
    :param date: String date with the format '%YYYY-%mm-%dd'."""
    return datetime.strptime(date, "%Y-%m-%d").date()

def daily_return(df, col):
    """Compute a Daily return of a dataframe.
    :param df: Dataframe strategy.
    :param col: Name of the column of df to compute daily returns."""
    df['daily_return'] = df[col]/df[col].shift(1)-1
    return df

def intersect_date(dfs):
    """Return a tuple of the Dataframes by keeping their common dates index.
    :param dfs: List of Dataframes."""
    if(len(dfs)<2):
        raise Exception('Should have at least 2 DataFrames.')
    else:
        ind = dfs[0].index
        for df in dfs[1:]:
            ind = sorted(set(ind).intersection(df.index))
        return [df.loc[ind] for df in dfs]

def dates_with_gap(dates, time):
    """Return a list with at least 'time' days between two consecutives dates.
    We look at the first date and keep the next one with at least 'time' days compared to the first one.
    We iterate this process with the keeped dates.
    :param dates: Series of dates in datetime format.
    :param time: Gap time in days."""
    dates = np.array(dates)
    current_date = dates[0]
    gap_dates = [current_date]
    while current_date<dates[-1]:
        try:           
            date = np.where(dates-current_date>=timedelta(days=time))[0][0]
            current_date = dates[date]
            gap_dates.append(current_date)
        except:
            current_date = dates[-1]
    return gap_dates

if __name__ == '__main__':
    spx_spot1 = import_data('SP', 'spot', '2006-01-01')
    spx_spot2 = import_data('SP', 'spot', '2010-01-01')
    str_to_datetime('2006-01-01')
    spx_spot1 = daily_return(spx_spot1, 'close')
    spx_spot1, spx_spot2 = intersect_date([spx_spot1, spx_spot2])
    dates = spx_spot1.index[[0,10,20,100,500]]
    dates_with_gap(dates, 30)