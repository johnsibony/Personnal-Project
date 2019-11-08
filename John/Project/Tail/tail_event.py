"""
List of fuctions to compute tail dates according to different method.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

from util import *
from password import *
from data import *
import pandas as pd
import numpy as np

def tail_by_quantile(df, roll_year=1, tp='down', qt=0.05):
    """Compute Tail date based on extreme (quantify by quantile) daily return.
    :param df: Dataframe index data with a column named 'close' containing prices.
    :param roll_year: Number of rolling year to apply quantile computation.
    :param tp: Type of tail event ('down' or 'up' for respectively negative tail or positive tail).
    :param qt: Quantile on the previous 'roll_year' year daily returns. If tp=='down', should be low quantile."""
    df['quantile'] = df['daily_return'].rolling(252*roll_year, min_periods=1).quantile(qt)
    if(tp=='down'):
    	df['tail_date'] = (df['daily_return']<=df['quantile']).astype(int)
    elif(tp=='up'):
    	df['tail_date'] = (df['daily_return']>=df['quantile']).astype(int)
    else:
    	raise KeyError("Argument 'tp' invalid. Should be 'down' or 'up' for respectively negative tail or positive tail.")
    tail_date = df[df['tail_date']==1].dropna().index
    return tail_date

def tail_by_sigma(df, roll_year=1, tp='down', sigma=3):
    """Compute Tail date based on daily returns moving away from their mean more than 'sigma' volatility.
    :param df: Dataframe index data with a column named 'close' containing prices.
    :param roll_year: Number of rolling year to apply mean and volatility computation on daily returns.
    :param tp: Type of tail event ('down' or 'up' for respectively negative tail or positive tail).
    :param sigma: Multiplicator coefficient of volatility."""
    df['mean'] = df['daily_return'].rolling(252*roll_year, min_periods=1).mean()
    df['daily_vol'] = df['daily_return'].rolling(252*roll_year, min_periods=1).std()
    if(tp=='down'):
        df['tail_date'] = ((df['daily_return']-df['mean'])<=-sigma*df['daily_vol']).astype(int)
    elif(tp=='up'):
        df['tail_date'] = ((df['daily_return']-df['mean'])>=sigma*df['daily_vol']).astype(int)
    tail_date = df[df['tail_date']==1].dropna().index 
    return tail_date

def tail_by_drawdown(df, threshold=0.05):
    """Compute a dataframe statistics of negative Tail event based on drawdown.
    What is a drawdown ? Take a price p1, and find the next price p2 such that p1=p2. On the period date of p1 and date of p2,
                         you compute the minimum value p3. The drawdown value is the return (p3-p1)/p1. In our cases, we only
                         considere drawdowns with less than minus threshold return.
    The returned Dataframe contains:  - The starting dates of a drawdown event.
                                      - The number of days between the starting date and the minimum peak value.
                                      - The date of the minimum peak value.
                                      - The negative drawdown return from the starting date value and the minimum peak value.
                                      - The number of days of the Drawdown.
                                      - The date of the end of the drawdown.
    :param df: Dataframe index data with a column named 'close' containing prices.
    :param threshold: Value needed to be considered such as a drawdown. If a return is less than minus 'threshold' then the event is a drawdown."""
    prices = list(df['close'].values)
    Drawdown_Start, Drawdown_Trading_Days, Drawdown_Min_Date, Cumulative_Drawdown_Return, Recovery_Trading_Days, Drawdown_End = [], [], [], [], [], []
    cols = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    current_index, current_price = 0, prices[0]
    while(current_index<len(prices)-2):
        res = [price>=current_price for price in prices[current_index+1:]]
        try: 
            max_boundary = current_index + res.index(True) + 1
        except:
            current_index += 1
            current_price = prices[current_index]
            continue
        min_boundary = current_index
        local_price = prices[min_boundary:max_boundary]
        local_min = min(local_price)
        if(local_min/current_price-1>-threshold):
            current_index += 1
        else:
            Drawdown_Start.append(current_index)
            Recovery_Trading_Days.append(max_boundary-min_boundary)
            Drawdown_End.append(max_boundary)
            Cumulative_Drawdown_Return.append(local_min/current_price - 1)
            local_min_ind = local_price.index(local_min)
            Drawdown_Trading_Days.append(local_min_ind)
            current_index = current_index + local_min_ind
            Drawdown_Min_Date.append(current_index)
        current_price = prices[current_index]
    df_drawdown = pd.DataFrame({'Start':df.index[Drawdown_Start], 'End':df.index[Drawdown_End], 'Time to Min':Drawdown_Trading_Days, 'Time to Recovery':Recovery_Trading_Days, 'Min Date':df.index[Drawdown_Min_Date], 'Min Return':Cumulative_Drawdown_Return})
    return df_drawdown

if __name__ == '__main__':
    spx_spot = import_data('SP', 'spot', '1990-01-01')
    spx_spot = daily_return(spx_spot, 'close')
    tail_by_quantile(spx_spot, roll_year=1, tp='up', qt=0.05)
    tail_by_sigma(spx_spot, roll_year=1, tp='down', sigma=3)
    tail_by_drawdown(spx_spot, threshold=0.05)
    