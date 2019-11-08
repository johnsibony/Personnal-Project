"""
Build Dataframe of trades (= 1 instrument).
"""

# Author: John Sibony <john.sibony@hotmail.fr>

from util import *
from data import *
from regime import MA_regime, DSTAT

def entry_dates(start, end, freq):
    """Compute the entry dates of trades of a strategy (dates to buy/sell contract).
    :param start: First date in datetime format.
    :param end: Last date in datetime format.
    :param freq: Frequency of the dates between 'start' and 'end' (see format allowed: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases)."""
    try:
        dates = pd.date_range(start, end, freq=freq).date
    except:
        raise KeyError('Invalid frequency. See the following link for valid frequency : https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases .')
    return dates

class Instrument:
    """Class to define an instrument. 
    An instrument is a set of trades/contracts opened at specific dates with specific features."""
    
    def __init__(self, data, entry_dates, dte, position, tp, end_contract, delta='', strike=''):
        """
        :param data: Dataframe data.
        :param entry_dates: Date of entry of the trades in datetime format.
        :param dte: Minimum number of life days of the contract.
        :param position: Binary integer (1 or -1 for respectively Long or Short position).
        :param tp: Type of contracts ('o' or 'f' for respectively option or future).
        :param end_contract: End way of the trades ('Roll' or 'Expiry' for respectively rolling or let to expiry the contracts).
        :param delta (Optional): Optional target delta for option contract.
        :param strike (Optional): Optional target strike for option contract."""
        if(tp=='o' and delta=='' and strike==''):
            raise KeyError('Must specify delta or strike target for option contract. If delta and strike re specified, delta argument will be used as target.')
        instrument = self.fit(data, entry_dates, dte, position, tp, end_contract, delta, strike)
        self.instrument = instrument
        
    @staticmethod
    def __find_maturity(df, date_start, dte):
        """Compute the maturity date with at least 'dte' days to expiry.
        :param df: Dataframe data.
        :param date_start: Beginning date of the contract in datetime format.
        :param dte: Minimum number of life days of the contract."""
        maturities = df.loc[[date_start], 'maturity'].values
        time_to_expiry = maturities - date_start
        try:
            selectioned_index = np.where(time_to_expiry>=timedelta(days=dte))[0][0]
        except:
            raise KeyError('No maturity find with at least {} time to expiry at the date {}.'.format(dte, date_start))
        maturity = maturities[selectioned_index]
        return maturity

    @staticmethod
    def __trades(entry_dates, maturities, end_contract):
        """Return a Dataframe with trades information (trade_id, dates, maturity).
        :param entry_dates: Date of entry of the trades in datetime format.
        :param maturities: Maturities of the trades.
        :param end_contract: End way of the trades ('Roll' or 'Expiry' for respectively rolling or let to expiry the contracts)."""
        trade_maturity, trade_dates, trade_id = [], [], []
        for i in range(0, len(entry_dates)-1):
            if(end_contract=='Roll'):
                dates = pd.date_range(entry_dates[i], entry_dates[i+1], freq='B').date
            elif(end_contract=='Expiry'):
                dates = pd.date_range(entry_dates[i], maturities[i], freq='B').date
            trade_maturity.append([maturities[i]]*len(dates))
            trade_dates.append(list(dates))
            trade_id.append([i]*len(dates))
        trade_maturity = list(chain(*trade_maturity))
        trade_dates = list(chain(*trade_dates))
        trade_id = list(chain(*trade_id))
        trades = pd.DataFrame(list(zip(trade_dates, trade_maturity, trade_id)), columns=['date', 'maturity', 'trade_id'])
        return trades

    @staticmethod
    def __filter_trade(trade, type_target, value):
        """Filter trades by 'type_target'. Function only applied for option contract.
        :param trade: Dataframe data.
        :param type_target: 'delta' or 'strike'.
        :param value: Value of the target."""
        date = trade.index[0]
        initial_options = trade.loc[date]
        initial_options = initial_options.reset_index()
        if(type_target=='delta'):
            selectioned_index = (initial_options['delta']-value).abs().sort_values().index[0]
        else:
            selectioned_index = (initial_options['strike']/initial_options['underlying']-value).abs().sort_values().index[0]
        selectioned_strike = initial_options.loc[selectioned_index]['strike']
        trade = trade[trade['strike']==selectioned_strike]
        return trade

    def __compute_pnl(self, df, delta, strike):
            if(delta):
                df = self.__filter_trade(df, 'delta', delta)
            elif(strike):
                df = self.__filter_trade(df, 'strike', strike)
            df.ix[0, 'first_trading_day'] = 1
            df['P&L_trade'] = df['position'] * (df['value'] - df['value'].shift(1)).fillna(0)
            return df

    def fit(self, data, entry_dates, dte, position, tp, end_contract, delta='', strike=''):
        """Compute the trades of the instrument.
        Remark : a simple way would be to find the contract to open for each entry dates (matching the dte, delta and end_contract arguemnt). Then,
                 to capture these contract into the DataFrame 'data' with the groupby entry_dates method. But for Python, this method is time consuming.
                 Indeed, it is more efficient to filter manually undesired contracts directly from the DataFrame 'data' by removing undesired dates. 
        :param data: Dataframe data.
        :param entry_dates: Date of entry of the trades in datetime format.
        :param dte: Minimum number of life days of the contract.
        :param position: Binary integer (1 or -1 for respectively Long or Short position).
        :param tp: Type of contracts ('o' or 'f' for respectively option or future).
        :param end_contract: End way of the trades ('Roll' or 'Expiry' for respectively rolling or let to expiry the contracts).
        :param delta (Optional): Optional target delta for option contract.
        :param strike (Optional): Optional target strike for option contract."""
        entry_dates = list(sorted(set(entry_dates).intersection(data.index)))
        data = data.rename(columns={'expiry_date': 'maturity', 'option_expiration': 'maturity', 'close':'value'})
        maturities = list(map(lambda x: self.__find_maturity(data, x, dte), entry_dates))
        date_info = self.__trades(entry_dates, maturities, end_contract)
        backtest = data
        backtest['position'], backtest['type'], backtest['first_trading_day'] = position, tp, 0
        if(tp=='o'):
            backtest['delta'] = backtest['position'] * backtest['delta']
            backtest['delta'] = pd.to_numeric(backtest['delta'])
        backtest['new_col'] = list(zip(backtest['maturity'], backtest.index))
        date_info['new_col'] = list(zip(date_info['maturity'], date_info['date']))
        backtest = backtest[backtest['new_col'].isin(date_info['new_col'])]
        backtest['count'] = backtest['new_col'].map(date_info.groupby('new_col').size())
        ids = backtest['new_col'].map(date_info.groupby('new_col').apply(lambda x: x['trade_id'].values))
        ids = list(chain(*ids))
        backtest = pd.DataFrame(np.repeat(backtest.values, backtest['count'].values, axis=0), columns=backtest.columns, index=np.repeat(backtest.index, backtest['count'].values))
        backtest['trade_id'] = ids
        groups = backtest.groupby('trade_id')
        backtest = [self.__compute_pnl(gp, delta, strike) for _,gp in groups]
        backtest = pd.concat(backtest)
        backtest = backtest.drop(['new_col', 'count'], axis=1)
        return backtest
        
    @staticmethod
    def _keep_trade(trade, dates):
        """Keep a trade if its entry date belongs to 'dates'.
        :param trade: Dataframe of a trade.
        :param dates: List of keeping dates in datetime format."""
        if(trade.index[0] in dates):
            return trade
        
    @staticmethod
    def _stop_date(trade, dates):
        """Stop a trade at the first date (included) occuring in 'dates' if it exists, keep the whole trade otherwise.
        :param trade: Dataframe of a trade.
        :param dates: List of stop dates in datetime format."""
        dates = sorted(trade.index.intersection(dates))
        if(dates):
            return trade[trade.index<=dates[0]]
        return trade
    
    @staticmethod
    def _stop_gain(trade, n_time):
        """Stop a trade when its value is 'n_time' times its entry value.
        :param trade: Dataframe of a trade.
        :param n_time: Multiplicator coefficient of the premium (entry value) of the trade."""
        return_premium = trade.value/trade.value.values[0]
        is_above = return_premium>=n_time
        if(sum(is_above)>0):
            last_date = trade[is_above].index[0]
            trade = trade.loc[trade.index<=last_date]
        return trade

    def adjust_instrument(self, fct_name, arg):
        """Filter the dates of each trades of an instrument.
        :param fct_name: Name of the function to apply on each trades of 'instrument' ('keep_trade' or 'stop_date' or 'stop_gain').
        :param arg: Argument of the function adjustment_fct (only one is accepted for both possible functions)."""
        instrument = self.instrument
        trades = instrument.groupby('trade_id')
        function = {'keep_trade':self._keep_trade, 'stop_date':self._stop_date, 'stop_gain':self._stop_gain}
        instrument = [function[fct_name](trade, arg) for _,trade in trades]
        instrument = pd.concat(instrument)
        self.instrument = instrument

    def pause(self, dates, time):
        """Take a pause of 'time' days after each dates of 'dates' : all of the new trade opened during this pause will be removed.
        :param dates: list of dates in datetime format.
        :param time: Number of calendar days for pauses."""
        dates = dates_with_gap(dates, time)
        instrument = self.instrument
        begin_trades_dates = instrument[instrument.first_trading_day==1]
        for date in dates:
            start_pause = date
            end_pause = date + timedelta(days=time)
            stop_period = pd.date_range(start_pause, end_pause, freq='D').date
            remove_dates = np.where(begin_trades_dates.index.isin(stop_period))[0]
            remove_trades = begin_trades_dates.iloc[remove_dates].trade_id
            instrument = instrument[~instrument.trade_id.isin(remove_trades)]
        self.instrument = instrument

if __name__ == '__main__':
    spx_put = import_data('SP', 'put', '2006-01-01')
    eom = entry_dates(spx_put.index[0], spx_put.index[-1], 'BM')
    opt = Instrument(spx_put, eom, 30, -1, 'o', 'Roll', 0.25)
    opt.adjust_instrument('stop_date', [opt.instrument.index[10]])
    opt.adjust_instrument('stop_gain', 3)
    opt.adjust_instrument('keep_trade', eom[0:10])
    opt.pause(opt.instrument.index[[50,21]], 10)