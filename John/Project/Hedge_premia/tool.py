"""
List of useful function for to compute HedgePremia strategies.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

import sys
sys.path.append('../Portfolio_construction')
from data import *
from regime import MA_regime, DSTAT

def change_trend(spx_spot):
    """Compute the date when trend changes (pass from bear to bull or bull to bear).
    :param spx_spot: SPX spot data."""
    bear = MA_regime(spx_spot.close,10,200,'bear')
    bull = MA_regime(spx_spot.close,10,200,'bull')
    trend = pd.Series(index=list(bear)+list(bull))
    trend.loc[bear], trend.loc[bull] = 0, 1
    trend = trend.sort_index()
    shift_trend = trend-trend.shift(1)
    change_bull = shift_trend[shift_trend==1].dropna().index
    change_bear = shift_trend[shift_trend==-1].dropna().index
    change = sorted(list(change_bull) + list(change_bear))
    return change

def search_sequence(arr, seq):
    """Return the indexes of the sequence(s) 'seq' in the array 'arr'.
    :param arr: 1D array of values.
    :param seq: 1D array of the sequence to find in 'arr'."""
    Na, Nseq = arr.size, seq.size
    r_seq = np.arange(Nseq)
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)
    if(M.any()>0):
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []

def search_dstat(spx_spot, seq, position, low_qt, high_qt):
    """ Returns dates of specific sequence value of DSTAT.
    Find the sequences 'seq' in the entries type array of DSTAT (1,0,-1). Then return the date corresponding at the 'position'th index of the sequence.
    :param spx_spot: SPX spot data.
    :param seq: List of sequence to match on Dstat (items must be -1 or 0 or 1 for respectively up or normal or down).
    :param position: Index of the list 'seq' to compute the date.
    :param low_qt: Lower quantile of DSTAT.
    :param high_qt: Higher quantile of DSTAT."""
    if(position>=len(seq)):
        n = len(seq)
        raise ValueError('Position argument should not be higher than the length of the sequence argument. Only {} positions are valid.'.format(list(range(n))))
    dstat = DSTAT(spx_spot, 1, 21, 63, 5, low_qt, high_qt).entry_type
    index = search_sequence(np.array(dstat),np.array(seq))
    index = [date_index for ind,date_index in enumerate(index) if ind%len(seq)==position]
    index = list(dstat.iloc[index].index)
    return index

def exit_dstat_down(spx_spot):
    """Compute exit dates for Dstat adjustment strategy.
    :param spx_spot: SPX spot data."""
    dstat = DSTAT(spx_spot, 1, 21, 63, 5, 0.05, 0.95)
    dstat_down = dstat.entry_type*(dstat.entry_type+1)/2
    exit_index = search_sequence(np.array(dstat_down),np.array([1,0]))
    exit_index = [date_index for ind,date_index in enumerate(exit_index) if ind%2==1]
    exit_index = [list(range(date_index,date_index+7)) for date_index in exit_index]
    exit_index = list(chain(*exit_index))
    dstat_down.iloc[exit_index] = 1
    index = search_sequence(np.array(dstat_down),np.array([1,0]))
    index = [date_index for ind,date_index in enumerate(index) if ind%2==0]
    index = list(dstat_down.iloc[index].index)
    return index

def next_date(current_date, dates):
    """Find the date following 'current_date' in dates.
    :param current_date: Current date in datetime format.
    :param dates: Series of dates in datetime format."""
    res = dates > current_date
    try:
        index = np.where(res)[0][0]
        return dates.iloc[index]
    except:
        return

def match_entry_exit(entries, exits):
    """Find successively start and end dates on list 'entries' and 'exits'.
    We find the start date of 'entries' which is just after the previous end date computed. Then we find the new end date on 'exit' which is just after the start date.
    We iterate this process until it is not possible to form a (start, end) couple.
    :param entries: List of entry dates.
    :param exits: List of exit dates."""
    entries, exits = pd.Series(entries), pd.Series(exits)
    dates, starts, ends = [], [], []
    end = entries.iloc[0]-timedelta(days=1)
    while(end<=exits.iloc[-1]):
        start = next_date(end, entries)
        end = next_date(start, exits)
        if(not start):
            break
        elif(not end):
            dates.append(pd.date_range(start, downs.index[-1], freq='B').date)
            break
        starts.append(start)
        ends.append(end)
        dates.append(pd.date_range(start, end, freq='B').date)
    dates = list(chain(*dates))
    return dates, starts, ends

def dates_stop_gain(trade, n_time):
    """Compute the first date of 'trade' when its value is 'n_time' times its entry value.
    :param trade: Dataframe of a trade.
    :param n_time: Multiplicator coefficient of the premium (entry value) of the trade."""
    return_premium = trade.value/trade.value.values[0]
    is_above = return_premium>=n_time
    if(sum(is_above)>0):
        last_date = trade[is_above].index[0]
        return last_date

if __name__ == '__main__':
    spx_spot = import_data('SP', 'spot', '2006-01-01')
    change_trend(spx_spot)
    search_sequence(np.array([0,1,2,3,4,5,1,2]), np.array([1,2]))
    search_dstat(spx_spot, np.array([1,0]), 1, 0.05, 0.95)
    exit_dstat_down(spx_spot)
    next_date(spx_spot.index[55], spx_spot.index)
    match_entry_exit(spx_spot.index[[0,1,2,6,8,45]], spx_spot.index[[1,2,7,10,15,25,44,89]])