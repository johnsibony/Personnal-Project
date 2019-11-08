"""
Build a Strategy (= set of instruments).
"""

# Author: John Sibony <john.sibony@hotmail.fr>

from util import *
from data import *
from instrument import *

class Strategy(Instrument):
    """Class to define a strategy.
    A strategy is a set of instrument(s) with additionnal features for each of its instrument(s)."""
    
    def __init__(self, name):
        """
        :param name: Name of the strategy."""
        self.name = name
        self.instruments = {}

    def get_instrument_ids(self):
        """Return the list of id(s) of instrument(s)."""
        return tuple(self.instruments.keys())

    def add_instrument(self, instrument_id, data, entry_dates, dte, position, tp, end_contract, delta='', strike=''):
        """Create an instrument for the strategy.
        :param df: Dataframe data.
        :param entry_dates: Date of entry of the trades in datetime format.
        :param dte: Minimum number of life days of the contract.
        :param position: Binary integer (1 or -1 for respectively Long or Short position).
        :param tp: Type of contracts ('o' or 'f' for respectively option or future).
        :param end_contract: End way of the trades ('Roll' or 'Expiry' for respectively rolling or let to expiry the contracts).
        :param delta (Optional): Target delta for option contract.
        :param strike (Optional): Target strike for option contract."""
        ids = self.get_instrument_ids()
        if(instrument_id in ids):
            raise ValueError('Instrument id already exist. Choose antother id')
        if(tp=='o' and delta=='' and strike==''):
            raise KeyError('Must specify delta or strike target for option contract.')
        print('Creation of Instrument {} of Strategy {}'.format(instrument_id, self.name))
        data = data.rename(columns={'expiry_date': 'maturity', 'option_expiration': 'maturity', 'close':'value'})
        entry_dates = list(sorted(set(entry_dates).intersection(data.index)))
        instrument = self.fit(data, entry_dates, dte, position, tp, end_contract, delta, strike)
        instrument['instrument_id'] = instrument_id
        instrument['strategy_id'] = self.name
        self.instruments[instrument_id] = instrument

    def __check_validity(self, instrument_id):
        """The user should specify different feature for each instrument by giving an instrument id. This function check the validity of the id provided.
        :param instrument_id: Id of the instrument given by the user."""
        ids = self.get_instrument_ids()
        if(instrument_id not in ids):
            raise KeyError('Instrument id {} not find. Only ids {} exist.'.format(instrument_id, ids))

    def delete_instrument(self, instrument_id):
        """Remove a specific instrument by its id.
        :param instrument_id: Id of the instrument given by the user."""
        self.__check_validity(instrument_id)
        del self.instruments[instrument_id]
        
    def adjust_instrument(self, instrument_id, tp, arg):
        """Filter the dates of each trades of an instrument.
        :param instrument_id: Id of the instrument.
        :param tp: Name of the function to apply on each trades of 'instrument' ('keep_trade' or 'stop_date' or 'stop_gain').
        :param arg: Argument of the function adjustment_fct (only one is accepted for both possible functions)."""
        self.__check_validity(instrument_id)
        instrument = self.instruments[instrument_id]
        trades = instrument.groupby('trade_id')
        function = {'keep_trade':self._keep_trade, 'stop_date':self._stop_date, 'stop_gain':self._stop_gain}
        instrument = [function[tp](trade, arg) for _,trade in trades]
        instrument = pd.concat(instrument)
        self.instruments[instrument_id] = instrument
        
    def pause(self, instrument_id, dates, time):
        """Take a pause of 'time' days after each dates of 'dates' : all of the new trade opened during this pause will be removed.
        :param dates: list of dates in datetime format.
        :param time: Number of calendar days for pauses."""
        dates = dates_with_gap(dates, time)
        self.__check_validity(instrument_id)
        instrument = self.instruments[instrument_id]
        begin_trades_dates = instrument[instrument.first_trading_day==1]
        for date in dates:
            start_pause = date
            end_pause = date + timedelta(days=time)
            stop_period = pd.date_range(start_pause, end_pause, freq='D').date
            remove_dates = np.where(begin_trades_dates.index.isin(stop_period))[0]
            remove_trades = begin_trades_dates.iloc[remove_dates].trade_id
            instrument = instrument[~instrument.trade_id.isin(remove_trades)]
        self.instruments[instrument_id] = instrument

    def type_investment(self, instrument_id, tp):
        """Type of investment for an instrument.
        :param instrument_id: Id of the instrument.
        :param tp: Type of investment ie way to compute quantity of contracts. Only 'underlying' (spot price) or 'value' (premium) argument are accepted.
                   Matematically : quantity=value_portfolio/type_investment."""
        self.__check_validity(instrument_id)
        columns = self.instruments[instrument_id].columns
        if(tp not in columns):
            raise ValueError('{} column is not defined for instrument id {}'.format(tp, instrument_id))
        instrument = self.instruments[instrument_id]
        instrument['type_investment'] = tp
        self.instruments[instrument_id] = instrument

    def get_weight_regime_format(self, instrument_id):
        """Return a Series in the valide format to fill the weights depending on regime of the specified instrument."""
        self.__check_validity()
        instrument = self.instruments[instrument_id]
        weights = pd.Series(index=sorted(set(instrument.index)))
        return weights

    def weights(self, instrument_id, weight):
        """Weight of an instrument.
        :param instrument_id: Id of the instrument to define the weight.
        :param weights: Float value (for constant weight) or Series filled by the Serie returned by the method 'get_weight_regime_format' (for weight depending on regime)"""
        self.__check_validity(instrument_id)
        instrument = self.instruments[instrument_id]
        if(isinstance(weight, pd.DataFrame)):
            entry_dates = instrument[instrument['first_trading_day']==1].index
            weight_entry_date = weight.loc[entry_dates]
            missing_date = weight_entry_date[weight_entry_date.isnull()].index
            if(missing_date.any()):
                raise TypeError('Certain dates for opening contract have no weights defined : {}'.format(missing_date))
        instrument['weight_instrument'] = weight

if __name__ == '__main__':
    spx_put = import_data('SP', 'put', '2006-01-01')
    spx_spot = import_data('SP', 'spot', '2006-01-01')
    opt = Strategy('my_strategy')
    eom = entry_dates(spx_put.index[0], spx_put.index[-1], 'BM')
    opt.add_instrument(1, spx_put, eom, 30, -1, 'o', 'Roll', 0.25)
    opt.type_investment(1, 'underlying')
    opt.weights(1,1)