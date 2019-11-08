"""
Statistics on negative Drawdowns.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

import sys
sys.path.append('../Portfolio_construction')
from portfolio import *
from util import *
from tail_event import *
import pandas as pd

def gap_drawdown(df):
    """Compute the gap time for each drawdown between its minimum peak value and the beginning of the next drawdown.
    :param df: Drawdown Dataframe returned by 'tail_by_drawdown' function."""
    current_start, next_start = df['Start'], df['Start'].shift(-1)
    time_to_min = pd.to_timedelta(df['Time to Min'], unit='D')
    current_start, next_start, time_to_min = current_start.iloc[:-1], next_start.iloc[:-1], time_to_min.iloc[:-1]
    gap = [next_start.iloc[ind]-(current_start.iloc[ind]+time_to_min.iloc[ind]) for ind in range(len(current_start))]
    return gap

def spx_trend_before_dd(df, days=10, thresholds=[0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15], display=True):
    """Return a Dataframe of the average number of positive S&P return for each 'days' days before drawdons defined by different thresholds.
    For each threshold, we compute the Drawdown starting dates. For each days before the starting dates, we compute the average number of positive S&P returns.
    :param df: Drawdown Dataframe returned by 'tail_by_drawdown' function.
    :param days: Number of days to be considred before drawdowns.
    :param thresholds: Value needed to be considered such as a drawdown. If a return is less than minus 'threshold' then the event is a drawdown.
    :param display: If True, plot the returned dataframe."""
    days = ['-{}day'.format(day) for day in range(days)]
    trend_return_on_dd = pd.DataFrame(index=thresholds, columns=days)
    plt.figure(figsize=(15,5))
    for threshold in thresholds:
        df_drawdown = tail_by_drawdown(df, threshold)
        dates = df_drawdown['Start'].values
        average_positive_return = []
        for day in range(len(days)):
            positive_return = []
            for date in dates:
                try:
                    positive_return.append(df[df.index<=date].iloc[-day-1]['daily_return']>=0)
                except:
                    pass
            average_positive_return.append(np.mean(positive_return))
        trend_return_on_dd.loc[threshold] = average_positive_return
        plt.plot(average_positive_return, label='-{}% DD'.format(100*threshold))
    plt.legend()
    plt.xlabel('Day before DD')
    plt.ylabel('% of positive return')
    plt.title('Average number of positive return before Drawdown.')
    plt.show()
    return trend_return_on_dd

def implied_realized_vol(df_spot, df_option, time_to_expiry=2, delta=-0.5, tp='after'):
    """Return the implied volatility and its corresponding realized volatility on the whole period.
    For each date of the dataframe 'df_option', we select the contract matching 'time_to_expiry' and 'delta' argument and return its implied volatility and life time.
    Then we compute the realized volatility on each of these dates on the period matching the life time of selected contract using 'df_spot' price.
    :param df_spot: Dataframe of spot price to compute realized vol.
    :param df_option: Dataframe option to compute implied vol.
    :param time_to_expiry: Length in months of options to compute implied vol.
    :param delta: Target delta to compute implied vol.
    :param tp: Type argument to compute realized vol ('before' or 'after' to respectively compute realized volatility before or after the 'start' date argument)."""
    def implied_vol(df):
        """For a date, return the implied volatility, date and maturity of the contract with at most 'time_to_expiry' days to expiry."""
        date = df.index[0]
        length = df['option_expiration']-date
        try:
            index = np.where(length<=timedelta(days=32*time_to_expiry))[0][-1]
        except:
            return np.nan
        maturity = df['option_expiration'].values[index]
        df = df[df['option_expiration']==maturity]
        df = df.reset_index()
        selectioned_index = (df['delta']-delta).abs().sort_values().index[0]
        iv = df.loc[selectioned_index]['iv']
        return iv, date, maturity
    def realized_vol(start, length):
        """Return the realized volatility during 'length' days, starting or ending at 'start' date depending on 'tp' argument.
        :param start: Referential date.
        :param length: Length period to compute realized volatility."""
        if(tp=='after'):
            df_filter = df_spot[(df_spot.index>=start) & (df_spot.index<=start+length)]
        elif(tp=='before'):
            df_filter = df_spot[(df_spot.index>=start-length) & (df_spot.index<=start)]
        else:
            raise KeyError("""Invalid type argument. Should be 'before' or 'after' to respectively compute realized volatility before or after 'start' date argument.""")
        rv = df_filter['daily_return'].std()*np.sqrt(252)
        return rv
    df = df_option.groupby(df_option.index).apply(implied_vol)
    df = df.dropna()
    ivols, dates, maturities = zip(*df)
    rvols = list(map(lambda x: realized_vol(x[0], x[1]-x[0]), list(zip(dates,maturities))))
    volatility = pd.DataFrame(list(zip(ivols,rvols)), index=dates, columns=['implied_vol', 'realized_vol']).dropna()
    return volatility

def index_around_dd(df_drawdown, index, days=21):
    """Display the evolution of 'index' serie 'days' days before and after drawdowns.
    Display also the Drawdown period with the evolution of 'index'.
    :param df_drawdown: Drawdown Dataframe returned by 'tail_by_drawdown' function.
    :param index: Serie of an index values."""
    filter_drawdown = df_drawdown[df_drawdown.Start>=index.index[0]]
    dates = filter_drawdown['Start'].values
    x_before, x_after = list(range(-days, 1)), list(range(0,days+1))
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    for ind,date in enumerate(dates):
        dates_before, dates_after = pd.date_range(end=date, periods=days+1, freq='B').date, pd.date_range(start=date, periods=days+1, freq='B').date
        plt.plot(x_before, np.array(index.loc[dates_before].values), color='red')
        plt.plot(x_after, np.array(index.loc[dates_after].values), color='green')
    plt.title('Indicator around Drawdown')
    plt.xlabel('Days centered around starting Drawdown')
    plt.ylabel('Indicator')
    plt.subplot(122)
    plt.plot(index.index[:-21], index[:-21], color='b', label='Indicator')
    [plt.axvspan(filter_drawdown['Start'].iloc[ind], filter_drawdown['Min Date'].iloc[ind], color='r', alpha=0.5) for ind in range(len(filter_drawdown))]
    plt.title('Indicator and Drawdown period')
    plt.legend()
    plt.show()
    
def trade_during_dd(df_drawdown, data, dte, position, tp, delta=''):
    """Display the evolution of price of contracts openned at the starting date of drawdowns.
    Useful to determine the type of strategy to choose (if drawdown dates are predictable).
    :param df_drawdown: Drawdown Dataframe returned by 'tail_by_drawdown' function.
    :param data: Dataframe data.
    :param dte: Minimum number of life days of the contract.
    :param position: Binary integer (1 or -1 for respectively Long or Short position).
    :param tp: Type of contracts ('o' or 'f' for respectively option or future).
    :param delta (Optional): Optional target delta for option contract."""
    def price_evolution(trade):
        """Display the evolution of price of a contract relatively to the entering price.
        :param trade: Trade on a drawdown date."""
        price_return = trade.value/trade.value.values[0]
        plt.plot(price_return.values)
    entry = df_drawdown['Start'].values
    strategy = Strategy('Price evolution')
    strategy.add_instrument(1, data, entry, dte, position, tp, 'Expiry', delta)
    plt.figure(figsize=(20, 5))
    strategy.instruments[1].groupby('trade_id').apply(price_evolution)
    plt.xlabel('Business days')
    plt.ylabel('Price(t) / Price(0)')
    plt.title('Price contract evolution compared to the opening price')
    plt.show()

if __name__ == '__main__':
    spx_spot = import_data('SP', 'spot', '1990-01-01')
    spx_spot = daily_return(spx_spot, 'close')
    vix_spot = import_data('VIX', 'spot', '1990-01-01')
    spx_put = import_data('SP', 'put', '1990-01-01', ('ES', 'EW3'))
    df_drawdown = tail_by_drawdown(spx_spot, threshold=0.075)
    gap_drawdown(df_drawdown)
    spx_trend_before_dd(spx_spot, days=10)
    volatility = implied_realized_vol(spx_spot, spx_put)
    index_around_dd(df_drawdown, vix_spot.close, days=21)
    index_around_dd(df_drawdown, spx_spot.daily_return, days=5)
    index_around_dd(df_drawdown, volatility.implied_vol, days=21)
    trade_during_dd(df_drawdown, spx_put, 35, -1, 'o', delta=0.25)