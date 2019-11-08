"""
Define regime dates based on indicators (useful to adjust trades based on regime dates).
"""

# Author: John Sibony <john.sibony@hotmail.fr>

from util import *
from data import *

def MA_regime(df, d1, d2, trend):
    """Compute bull or bear dates based on Moving Average.
    :param df: Series of spot.
    :param d1: Number of the shortest period of rolling business days.
    :param d2: Number of the longest period of rolling business days.
    :param trend: Name of the trend ('bull' or 'bear' for respectively Bull or Bear market regime)."""
    if(trend=="bull"):
        df_binary = (df.rolling(d1).mean()>=df.rolling(d2).mean())
    elif(trend=="bear"):
        df_binary = (df.rolling(d1).mean()<df.rolling(d2).mean())
    else:
        raise KeyError('''Non valid trend argument. Should be 'bull' or 'bear' ''')
    date = df_binary[df_binary==True].dropna().index
    return date

def vix_regime(df, low, high):
    """Compute dates of Vix's spot price belonging to the interval [low,high].
    :param df: Series of Vix spot.
    :param low: Lower bound.
    :param high: Upper bound."""
    df_binary = (df<high) & (df>low)
    date = df_binary[df_binary==True].dropna().index
    return date

def hvol_regime(df, days, low, high):
    """Compute dates of historic volatility prices belonging to the interval [low,high].
    :param df: Series of spot prices.
    :param days: Number of business rolling days to compute volatility.
    :param low: Lower bound.
    :param high: Upper bound."""
    if(low>high):
        raise KeyError('Low bound argument must be inferior to high bound argument')
    daily_return = df/df.shift(1)-1
    vol = daily_return.rolling(days).std() * np.sqrt(252) * 100
    df_binary = (vol<high) & (vol>low)
    #df_binary = df_binary.shift(1).dropna()
    date = df_binary[df_binary==True].dropna().index
    return date

def DSTAT(df, ma1, ma2, rv, roll_year, low_qt, high_qt):
    """Compute DSTAT indicator with its region regime.
    Region regime are defined by extreme values (quantify by 'low_qt' and 'high_qt' quatile) of DSTAT indicator.
    Region -1, 0, 1 corresponds respectively to extreme high, normal and extreme low DSTAT values.
    Mathematically : DSTAT = (MA(ma1)-MA(ma2)) / (MA(ma2)*VOL(rv)).
    :param df: Dataframe of spot price with column named 'close'.
    :param ma1: Number of business days for the MA1 indicator (generally 'ma1'=1 ie spot price).
    :param ma2: Number of business days for the MA2 indicator. 
    :param rv: Number of business days for the VOL indicator. 
    :param roll_year: Number of rolling year to compute extreme DSTAT quantile on each date.
    :param low_qt: Lower quantile of DSTAT.
    :param high_qt: Upper quantile of DSTAT."""
    df = daily_return(df, 'close')
    MA1 = df['close'].rolling(ma1).mean()
    MA2 = df['close'].rolling(ma2).mean()
    RV = df['daily_return'].rolling(rv).std() * np.sqrt(252)
    df['dstat'] = (MA1 - MA2) / (MA2 * RV)
    df['dstat_long_lev'] = df['dstat'].rolling(252*roll_year, min_periods=1).quantile(low_qt)
    df['dstat_long'] = df['dstat']<=df['dstat_long_lev']
    df['dstat_short_lev'] = df['dstat'].rolling(252*roll_year, min_periods=1).quantile(high_qt)
    df['dstat_short'] = df['dstat']>=df['dstat_short_lev']
    df['entry_type'] = df['dstat_long'].astype(int) - df['dstat_short'].astype(int)
    df = df[['dstat','entry_type']].dropna()
    return df

def DSTAT_regime(df, ma1, ma2, rv, roll_year, low_qt, high_qt, trend):
    """Compute DSTAT dates belonging to 'trend' regime.
     Mathematically : DSTAT = (MA1-MA2) / (MA*ROLLING_VOL).
    :param df: Dataframe of spot price with column named 'close'.
    :param ma1: Number of business days for the MA1 indicator. 
    :param ma2: Number of business days for the MA2 indicator. 
    :param rv: Number of business days for the ROLLING_VOL indicator. 
    :param roll_year: Number of rolling year to compute extreme DSTAT values on each date.
    :param low_qt: Lower quantile of DSTAT.
    :param high_qt: Upper quantile of DSTAT.
    :param trend: Integer of the DSTAT trend (1,0 or -1 for respecively extreme low, normal or extreme high DSTAT values)."""
    dstat = DSTAT(df, ma1, ma2, rv, roll_year, low_qt, high_qt)
    try:
        date = dstat[dstat.entry_type==trend].dropna().index
    except:
        raise KeyError('''Non valid trend argument. Should be 1,0 or -1 for respectively extreme low, normal or extreme high value of DSTAT. ''')
    return date

def plot_DSTAT_signal_by_year(df, ma1, ma2, rv, roll_year, low_qt, high_qt, trend):
    """Plot the number of DSTAT in 'trend' DSTAT region for each year.
    Mathematically : DSTAT = (MA1-MA2) / (MA*ROLLING_VOL).
    :param df: Dataframe of spot price with column named 'close'.
    :param ma1: Number of business days for the MA1 indicator. 
    :param ma2: Number of business days for the MA2 indicator. 
    :param rv: Number of business days for the ROLLING_VOL indicator. 
    :param roll_year: Number of rolling year to compute extreme DSTAT values on each date.
    :param low_qt: Lower quantile of DSTAT.
    :param high_qt: Upper quantile of DSTAT.
    :param trend: Integer of the DSTAT trend (1,0 or -1 for respecively extreme low, normal or extreme high DSTAT values)."""
    signal = DSTAT_regime(df, ma1, ma2, rv, roll_year, low_qt, high_qt, trend)
    signal = list(signal.map(lambda x: x.year))
    year = list(set(signal))
    occurrence = [signal.count(y) for y in year]
    plt.bar(year, occurrence)
    plt.title('Number of short signal')
    plt.show()

def plot_DSTAT_signal_duration(df, ma1, ma2, rv, roll_year, low_qt, high_qt, trend):
    """Plot the histogram of the duration in business days of 'trend' DSTAT signal.
    Mathematically : DSTAT = (MA1-MA2) / (MA*ROLLING_VOL).
    :param df: Dataframe of spot price with column named 'close'.
    :param ma1: Number of business days for the MA1 indicator. 
    :param ma2: Number of business days for the MA2 indicator.
    :param rv: Number of business days for the ROLLING_VOL indicator.
    :param roll_year: Number of rolling year to compute extreme DSTAT values on each date.
    :param low_qt: Lower quantile of DSTAT.
    :param high_qt: Upper quantile of DSTAT.
    :param trend: Integer of the DSTAT trend (1,0 or -1 for respecively extreme low, normal and extreme high DSTAT values)."""
    dstat = DSTAT(df, ma1, ma2, rv, roll_year, low_qt, high_qt)
    if(trend==1):
        signal = (1 + dstat['entry_type']) * dstat['entry_type'] / 2
    elif(trend==0):
        signal = (1 + dstat['entry_type']) * (1 - dstat['entry_type'])
    elif(trend==-1):
        signal = (1 - dstat['entry_type']) * dstat['entry_type'] / 2
    else:
        raise KeyError('''Non valid trend argument. Should be 1,0 or -1 for respectively 'down','normal' or 'up' ''')
    signal = signal.cumsum().value_counts()
    signal = pd.Series(sorted(signal[signal>1].index))
    signal = (signal - signal.shift(1)).dropna().values
    plt.hist(signal)
    plt.title('Signal Duration')
    plt.show()

def plot_DSTAT_signal_gap(df, ma1, ma2, rv, roll_year, low_qt, high_qt, trend):
    """Plot the histogram of the gap in business days between two consecutives 'trend' DSTAT signal.
    Mathematically : DSTAT = (MA1-MA2) / (MA*ROLLING_VOL).
    :param df: Dataframe of spot price with column named 'close'.
    :param ma1: Number of business days for the MA1 indicator. 
    :param ma2: Number of business days for the MA2 indicator. 
    :param rv: Number of business days for the ROLLING_VOL indicator. 
    :param roll_year: Number of rolling year to compute extreme DSTAT values on each date.
    :param low_qt: Lower quantile of DSTAT.
    :param high_qt: Upper quantile of DSTAT.
    :param trend: Integer of the DSTAT trend (1,0 or -1 for respecively extreme low, normal and extreme high DSTAT values)."""
    signal = DSTAT_regime(df, ma1, ma2, rv, roll_year, low_qt, high_qt, trend)
    gap_days = pd.Series(signal).shift(-1) - signal
    gap_days = gap_days.dropna()
    plt.hist([date.days for date in gap_days])
    plt.title('Gap in days')
    plt.show()

if __name__ == '__main__':
    spx_spot = import_data('SP', 'spot', '2006-01-01')
    vix_spot = import_data('VIX', 'spot', '2006-01-01')
    MA_regime(spx_spot.close, 10, 200, 'bull')
    vix_regime(vix_spot.close, 13, 25)
    hvol_regime(spx_spot.close, 252, 0, 50)
    plot_DSTAT_signal_by_year(spx_spot, 1, 21, 63, 5, 0.05, 0.95, 1)
    plot_DSTAT_signal_duration(spx_spot, 1, 21, 63, 5, 0.05, 0.95, 1)
    plot_DSTAT_signal_gap(spx_spot, 1, 21, 63, 5, 0.05, 0.95, 1)