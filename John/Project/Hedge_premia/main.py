"""
HedgePremia strategies.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

import sys
sys.path.append('../Portfolio_construction')
from data import *
from portfolio import *
from regime import MA_regime
from tool import *

def delta_neutral(spx_put, spx_future, spx_spot):
    """Contract: Short SPX put 25Δ  (>35dte) / Short SPX future (>35dte).
    Entry: Every last business day of the month + at Dstat Down event.
    Exit: Roll.
    Weight: 1 / 0.25
    :param spx_put: SPX put data.
    :param spx_future: SPX future data.
    :param spx_spot: SPX spot data."""
    entry_dstat = search_dstat(spx_spot, [0,1,1], 2, 0.05, 0.95)
    entry_trend = change_trend(spx_spot)
    entry_eom = entry_dates(spx_put.index[0], spx_put.index[-1], 'BM')
    entry = list(entry_dstat)+list(entry_trend)+list(entry_eom)
    delta_neutr = Strategy('Delta neutral')
    delta_neutr.add_instrument(1, spx_put, entry, 35, -1, 'o', 'Roll', 0.25)
    delta_neutr.add_instrument(2, spx_future, entry_eom, 35, -1, 'f', 'Roll')
    delta_neutr.type_investment(1, 'underlying')
    delta_neutr.type_investment(2, 'underlying')
    delta_neutr.weights(1,1)
    delta_neutr.weights(2,0.25)
    return delta_neutr

def trend_adjustment(spx_future, spx_spot):
    """Contract: Short SPX future (>35dte).
    Entry: Every last business day of the month + when trend (bull/bear) changes.
    Exit: Roll.
    Weight: 0.25 bull and 0.75 bear.
    :param spx_future: SPX future data.
    :param spx_spot: SPX spot data."""
    entry_trend = change_trend(spx_spot)
    entry_eom = entry_dates(spx_future.index[0], spx_future.index[-1], 'BM')
    entry = list(entry_trend)+list(entry_eom)
    trend_adj = Strategy('Trend adjustment')
    trend_adj.add_instrument(1, spx_future, entry, 35, -1, 'f', 'Roll')
    trend_adj.type_investment(1, 'underlying')
    bear, bull = MA_regime(spx_spot.close,10,200,'bear'), MA_regime(spx_spot.close,10,200,'bull')
    weights = pd.Series(0.75, index=list(bear)+list(bull))
    weights.loc[bull] = 0.25
    trend_adj.weights(1,weights)
    return trend_adj

def dstat_adjustment(spx_future, spx_spot):
    """Contract: Short SPX future (>35dte).
    Entry: Every last business day of the month + when trend (bull/bear) changes + at Dstat Down event.
    Exit: Roll.
    Weight: 0.25 bull and 0.75 bear.
    :param spx_future: SPX future data.
    :param spx_spot: SPX spot data."""
    entry_trend = change_trend(spx_spot)
    entry_eom = entry_dates(spx_future.index[0], spx_future.index[-1], 'BM')
    entry_dstat, exit_dstat = search_dstat(spx_spot, [0,1,1], 2, 0.05, 0.95), exit_dstat_down(spx_spot)
    down_dates, entry_dstat, exit_dstat = match_entry_exit(entry_dstat, exit_dstat)
    entry = list(entry_trend)+list(entry_eom)+list(entry_dstat)
    entry = sorted(set(entry).intersection(down_dates))
    dstat_adj = Strategy('DSTAT adjustment')
    dstat_adj.add_instrument(1, spx_future, entry, 35, 1, 'f', 'Roll')
    dstat_adj.adjust_instrument(1, 'stop_date', exit_dstat)
    dstat_adj.type_investment(1, 'underlying')
    bear, bull = MA_regime(spx_spot.close,10,200,'bear'), MA_regime(spx_spot.close,10,200,'bull')
    weights = pd.Series(0.75, index=list(bear)+list(bull))
    weights.loc[bull] = 0.25
    dstat_adj.weights(1,weights)
    return dstat_adj

def tail_hedging(spx_put, spx_spot):
    """Contract : Long SPX put -0.05Δ (>35dte).
    Entry : Pass from Dstat up to Dstat down + 
    Exit: Roll + Pass from Dstat down to Dstat up + 2x stop gain with pause of 180days.
    Weight : 0.01 (based on premium value, not underlying).
    :param spx_put: SPX put data.
    :param spx_spot: SPX spot data."""
    bull = MA_regime(spx_spot.close,10,200,'bull')
    entry_dstat, exit_dstat = search_dstat(spx_spot, [-1,0], 1, 0.1, 0.9), search_dstat(spx_spot, [0,-1], 1, 0.1, 0.9)
    entry = set(entry_dstat).intersection(bull)
    tail = Strategy('Tail')
    tail.add_instrument(1, spx_put, entry, 35, 1, 'o', 'Expiry', -0.05)
    tail.adjust_instrument(1, 'stop_date', exit_dstat)
    dates = list(tail.instruments[1].groupby('trade_id').apply(lambda x: dates_stop_gain(x, 2)).dropna().values)
    dates.remove(str_to_datetime('2010-04-27'))
    tail.pause(1, dates, 180)
    tail.adjust_instrument(1, 'stop_gain', 2)
    tail.type_investment(1, 'value')
    tail.weights(1,0.01)
    return tail

def TY(bond_future, spx_spot):
    """Contract : Long 10Y TY bond future (>35dte).
    Entry : Every last business day of the month + when trend (bull/bear) changes.
    Exit: Roll.
    Weight : 0.75 bull and 0.25 bear.
    :param bond_future: 10Y TY future data.
    :param spx_spot: SPX spot data."""
    entry_trend = change_trend(spx_spot)
    entry_eom = entry_dates(bond_future.index[0], bond_future.index[-1], 'BM')
    entry = sorted(list(entry_trend)+list(entry_eom))
    ty = Strategy('TY')
    ty.add_instrument(1, bond_future, entry, 35, 1, 'f', 'Roll')
    ty.type_investment(1, 'value')
    bear, bull = MA_regime(spx_spot.close,10,200,'bear'), MA_regime(spx_spot.close,10,200,'bull')
    weights = pd.Series(0.25, index=list(bear)+list(bull))
    weights.loc[bull] = 0.75
    ty.weights(1,weights)
    return ty

def benchmark(spx_future):
    """Contract : Short SPX future (>90dte).
    Entry : Every last business day of the month.
    Exit: Roll.
    Weight : 1.
    :param spx_future: SPX future data."""
    eom = entry_dates(spx_future.index[0], spx_future.index[-1], 'BM')
    bench = Strategy('Benchmark')
    bench.add_instrument(1, spx_future, eom, 30*3, -1, 'f', 'Roll')
    bench.type_investment(1, 'underlying')
    bench.weights(1,1)
    return bench

def hedge_premia(spx_put, spx_future, bond_future, spx_spot, weights={'Delta neutral':1, 'Trend adjustment':1, 'DSTAT adjustment':1, 'Tail':1, 'TY':1}):
    """HedgePremia portfolio.
    :param spx_put: SPX put data.
    :param spx_future: SPX future data.
    :param bond_future: 10Y TY future data.
    :param spx_spot: SPX spot data.
    :param weights: weights strategies (by regime is also allowed)."""
    strategy1 = delta_neutral(spx_put, spx_future, spx_spot)
    strategy2 = trend_adjustment(spx_future, spx_spot)
    strategy3 = dstat_adjustment(spx_future, spx_spot)
    strategy4 = tail_hedging(spx_put, spx_spot)
    strategy5 = TY(bond_future, spx_spot)
    hp = Portfolio(strategy1, strategy2, strategy3, strategy4, strategy5)
    hp.weights(weights)
    hp.fit()
    return hp

if __name__ == '__main__':
    spx_spot = import_data('SP', 'spot', '1975-01-01')
    spx_call = import_data('SP', 'call', '2006-01-01')
    spx_put = import_data('SP', 'put', '2006-01-01', freq=('EW3', 'ES'))
    spx_future = import_data('SP', 'future', '2006-01-01')
    spx_future['underlying'] = spx_spot['close']
    bond_future = import_data('10Ybond', 'future', '2006-01-01')
    hp = hedge_premia(spx_put, spx_future, bond_future, spx_spot)
    hp.monthly_return()