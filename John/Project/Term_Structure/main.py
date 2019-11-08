"""
Term structure strategies.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

import sys
sys.path.append('../Portfolio_construction')
from portfolio import *
from data import *
from tool import *

def term_structure_arbitrage(contango_date, backwardation_date, data, tp, delta=''):
    """Contract: Long 'data' index parameter when backwardation / Short 'data' index parameter when contango.
    Entry: Every last business day of the month.
    Exit: Roll.
    Weight: 1 / 1
    :param contango_date: List of date of contango structure.
    :param backwardation_date: List of date of contango backwardation structure.
    :param data: Data index (generally, VIX future or VIX call).
    :param tp: Type of contracts ('o' or 'f' for respectively option or future).
    :param delta (Optional): target delta for Long option contract."""
    eom = entry_dates(data.index[0], data.index[-1], 'BM')
    term_structure = Strategy('Term Structure')
    if(tp=='o'):
        term_structure.add_instrument(1, data, eom, 35, -1, tp, 'Roll', -delta)
        term_structure.add_instrument(2, data, eom, 35, 1, tp, 'Roll', delta)
    else:
        term_structure.add_instrument(1, data, eom, 35, -1, tp, 'Roll')
        term_structure.add_instrument(2, data, eom, 35, 1, tp, 'Roll')
    term_structure.adjust_instrument(1, 'keep_trade', contango_date)
    term_structure.adjust_instrument(2, 'keep_trade', backwardation_date)
    term_structure.type_investment(1, 'underlying')
    term_structure.type_investment(2, 'underlying')
    term_structure.weights(1,1)
    term_structure.weights(2,1)
    term_structure = Portfolio(term_structure)
    term_structure.weights({'Term Structure':1})
    term_structure.fit()
    return term_structure

def volatility_carry_strategy(vix_future, vxx):
    """ See: http://jonathankinlay.com/2019/07/developing-volatility-carry-strategy/
    Does not work.
    Contract: Long VIX future / Short VXX index.
    Entry: Every last business day of the month.
    Exit: Roll.
    Weight: 1 / 1
    :param vix_future: VIX future data.
    :param vxx: VXX ETN prices."""
    eom = entry_dates(vxx.index[0], vxx.index[-1], 'BM')
    carry = Strategy('Carry')
    carry.add_instrument(1, vxx, eom, 20, -1, 'f', 'Roll')
    carry.add_instrument(2, vix_future, eom, 20, 1, 'f', 'Roll')
    carry.type_investment(1, 'underlying')
    carry.type_investment(2, 'underlying')
    carry.weights(1,1)
    carry.weights(2,1)
    carry = Portfolio(carry)
    carry.weights({'Carry':1})
    carry.fit()
    return carry

if __name__ == '__main__':
    vix_spot = import_data('VIX', 'spot', '1975-01-01')    
    vix_future = import_data('VIX', 'future', '2006-01-01')
    vix_future['underlying'] = vix_spot['close']
    vix_call = import_data('VIX', 'call', '2006-01-01')
    structure = np.sign(vix_future['close'].groupby(vix_future.index).apply(term_structure))
    contango_date = structure[structure==1].index
    backwardation_date = structure[structure==-1].index
    fut_portfolio = term_structure_arbitrage(contango_date, backwardation_date, vix_future, 'f')
    opt_portfolio = term_structure_arbitrage(contango_date, backwardation_date, vix_call, 'o', delta=0.05)
    
    link_engine = get_link_engine()
    query = '''select date,close from data_ohlc.nysearca_vxx where date >= '2006-01-01' '''
    vxx =  extraction_data(link_engine, query)
    vxx.sort_values(['date'], inplace=True)
    vxx = vxx.set_index("date")
    vxx['maturity'] = list(map(front_end_month, vxx.index))
    vxx['underlying'] = vxx['close']
    portfolio = volatility_carry_strategy(vix_future, vxx)
    portfolio.plot_performance()