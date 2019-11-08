"""
Tail portfolio.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

import sys
sys.path.append('../Portfolio_construction')
from portfolio import *
from regime import MA_regime

def tail_vix_future(vix_spot, vix_future):
    """Contract: Short VIX future (>35dte).
    Entry: Every friday if VIX spot >= 15 and bear market.
    Exit: Every friday if VIX spot < 15 or bull market.
    Weight: 1 (based on VIX future price, not VIX spot).
    :param vix_spot: Vix spot data.
    :param vix_future: Vix future data."""
    bear, bull = MA_regime(vix_spot.close,5,100,'bear'), MA_regime(vix_spot.close,5,100,'bull')
    entry_vix, exit_vix = vix_spot[vix_spot.close>=15].dropna().index, vix_spot[vix_spot.close<15].dropna().index
    weekly_friday = pd.date_range(vix_future.index[0], vix_future.index[-1], freq='W-FRI').date
    entry, exit = set(bear).intersection(entry_vix, weekly_friday), set(set(bull).union(exit_vix)).intersection(weekly_friday)
    tail_vix = Strategy('Tail')
    tail_vix.add_instrument(1, vix_future, entry, 35, -1, 'f', 'Roll')
    tail_vix.adjust_instrument(1, 'stop_date', exit)
    tail_vix.type_investment(1, 'underlying')
    tail_vix.weights(1,1)
    tail_vix = Portfolio(tail_vix)
    tail_vix.weights({'Tail':1})
    tail_vix.fit()
    return tail_vix

if __name__ == '__main__':
    vix_spot = import_data('VIX', 'spot', '1975-01-01')    
    vix_future = import_data('VIX', 'future', '2006-01-01')
    vix_future['underlying'] = vix_future['close'] #Quantity based on future prices not spot price!
    portfolio = tail_vix_future(vix_spot, vix_future)
    portfolio.plot_performance()
    portfolio.monthly_return()