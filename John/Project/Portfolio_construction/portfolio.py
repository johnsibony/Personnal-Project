"""
Build a portfolio by combining Strategies.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

from util import *
from data import *
from strategy import *
from regime import MA_regime
import math
import quandl #module for risk free rate
import warnings

class Portfolio:
    """Class to define a portfolio.
    A portfolio is a set of strategies."""

    ####################################################### Backtest #######################################################
    
    def __init__(self, *strategies):
        """
        :param strategies: Tuple of Strategy object."""
        if(not strategies):
            raise TypeError('No strategy defined.')
        for strategy in strategies:
            for instrument_id,instrument in strategy.instruments.items():
                if('weight_instrument' not in instrument.columns):
                    raise TypeError('No weight has been defined for instrument id {} of strategy id {}.'.format(instrument_id, strategy_id))
                if('type_investment' not in instrument.columns):
                    raise TypeError('No type of investment has been defined for instrument id {} of strategy id {}.'.format(instrument_id, strategy_id))
        self.strategies = {strategy.name:strategy for strategy in strategies}
        self.portfolio = pd.DataFrame()

    def get_strategy_ids(self):
        """Return the list of id(s) of instrument(s)."""
        return tuple(self.strategies.keys())

    def get_weight_regime_format(self):
        """Return a DataFrame in the valide format to fill the weights strategies depending on regime."""
        index = [instrument.index for strategy in self.strategies.values() for instrument in strategy.instruments.values()]
        index = list(chain(*index))
        weights = pd.DataFrame(index=sorted(set(index)), columns=self.strategies.keys())
        return weights
        
    def weights(self, weights):
        """Weights of each strategies.
        :param weights : Dict for constant weights (keys: id strategy, values: weight of strategy) or Dataframe of the weights by dates index for each strategies. 
                         Each column is a strategy with the name of a column matching the id strategy."""
        strategy_ids = tuple(self.strategies.keys())
        if(isinstance(weights, dict)):
            weight_ids = weights.keys()
        elif(isinstance(weights, pd.DataFrame)):
            weight_ids =  weights.columns
        else:
            raise TypeError('Argument weight should be a dictionnary or a DataFrame.')
        for weight_id in weight_ids:
            if(weight_id not in strategy_ids):
                raise KeyError('Id {0} does not exist. Only ids {1} are available'.format(weight_id, strategy_ids))
        for strategy_id,strategy in self.strategies.items():
            for instrument_id,instrument in strategy.instruments.items():
                instrument['weight_strategy'] = weights[strategy_id]
                instrument['weight'] = instrument['weight_strategy'] * instrument['weight_instrument']

    @staticmethod
    def __compute_delta(df_by_date):
        """This function compute the delta of the portfolio for a date.
        :param df_by_date: Dataframe of trades for a date."""
        total_qt = df_by_date['quantity'].sum()
        delta_opt = np.matmul(df_by_date['quantity'] , df_by_date['delta'])
        fut =  df_by_date[df_by_date['type']=='f']
        delta_fut = np.matmul(fut['quantity'] , fut['position'])
        return (delta_opt + delta_fut) / total_qt

    def fit(self):
        """This function computes the final backtest by combining one or several strategies of instruments.
        It allows to change weights of strategies depending on regime condition. This switch would occur only when openning a new contract/trade."""
        def compute_performance(df_by_date):
            """This function compute the performance of the portfolio for a date.
            :param df_by_date: Dataframe of trades for a date."""
            new_contract = df_by_date[df_by_date['first_trading_day']==1]
            ids_new_contract = list(zip(new_contract['trade_id'].values, new_contract['strategy_id'].values, new_contract['instrument_id'].values))
            if(len(ids_new_contract)>0):
                qt_key = ids_new_contract
                qt_value = new_contract['weight'] * b[0] / new_contract.apply(lambda x: x[x.type_investment], axis=1)
                qt.update(list(zip(qt_key, qt_value)))
            old_contract = df_by_date[df_by_date['first_trading_day']==0]
            if(not old_contract.empty):
                ids_old_contract = list(zip(old_contract['trade_id'].values, old_contract['strategy_id'].values, old_contract['instrument_id'].values))
                old_contract['Daily_P&L'] = np.dot(np.array([qt[i] for i in ids_old_contract]),old_contract['P&L_trade'])
                old_contract['base'] = b[0] + old_contract['Daily_P&L']
                b[0] = old_contract['base'].values[0]
            else:
                return
            return old_contract
        df = pd.concat([instrument for strategy in self.strategies.values() for instrument in strategy.instruments.values()])
        df = df.sort_index()
        df['base'], b, qt = 0, np.ones(1), {}
        groups = df.groupby(df.index)
        df = [compute_performance(gp) for _,gp in groups]
        df = pd.concat(df)
        bankrupt_dates = df[df.base<0].dropna()
        if(not bankrupt_dates.empty):
            bankrupt_date = bankrupt_dates.index[0]
            df = df.loc[df.index<bankrupt_date]
            warnings.warn('\n----- Negative value of portfolio. Backtesting stopped at date {} -----'.format(bankrupt_date))
        df = daily_return(df, 'base')
        df['quantity'] = df.set_index(['trade_id', 'strategy_id', 'instrument_id']).index.map(qt)
        df = df.fillna(0)
        tp = sorted(set(df['type']))
        if('o' in tp):
            df['delta'] = df.groupby(df.index).apply(self.__compute_delta)
        col_name = ['base', 'daily_return', 'strategy_id', 'instrument_id', 'type', 'position', 'trade_id', 'weight', 'weight_instrument', 'weight_strategy', 'weight', 'quantity', 'maturity', 'underlying'] + ('o' in tp)*['delta']
        df = df[col_name]
        self.portfolio = df

    ####################################################### Performance methods #######################################################

    def extract_daily_return(self):
        """Extract the Daily return of a DataFrame portfolio. The Dataframe Strategy df can have duplicates date indexes (when there is several trade at the same date).
        This function retrieve daily return of df without duplicate (one return by date)."""
        df = self.portfolio
        daily_return = df['daily_return'].groupby(df.index).first()
        return daily_return

    def extract_monthly_return(self):
        """Extract the Monthly returns of a DataFrame portfolio. The 'extract_daily_return' function is first called. Then Monthly returns are extracted from it.
        We then use a function 'reformat_index' to change into a more convenient format the date indexes of the monthly return."""
        def reformat_index(index):
            """ Change the index format of the Monthly returns dataframe.
            Pass from the MultiIndex format [%YYYY,%mm] to the index format %YYYY-%mm-%01 """
            if(index[1]<10):
                return str_to_datetime(str(index[0])+'-0'+str(index[1])+'-01')
            else:
                return str_to_datetime(str(index[0])+'-'+str(index[1])+'-01')
            return monthly_return
        df = self.portfolio
        daily_return = df['daily_return'].groupby(df.index).first()
        monthly_return = daily_return.groupby([daily_return.index.map(lambda x: x.year), daily_return.index.map(lambda x: x.month)]).apply(self.__geometric_return)
        monthly_return.index = [reformat_index(index) for index in monthly_return.index]
        return monthly_return

    @staticmethod
    def __geometric_return(daily_return):
        """Extract the rate of a period of daily returns. If the daily returns of a month is given, the function computes its corresponding monthly rate.
        Matematically : find r such that  :  1+r=PROD(1+daily_return).
        :param daily_return: Daily return on a given period."""
        return (1+daily_return).prod()-1

    def VAR(self, quantile):
        """Compute the historical Value At Risk (VAR) based on daily_return.
        :param quantile: Quantile of the words Daily returns."""
        daily_return = self.extract_daily_return()
        daily_return = list(daily_return.dropna())
        var = np.percentile(daily_return, quantile) * 100
        var = round(var, 2)
        return var

    def max_drawdown(self):
        """Compute the Maximum Drawdown. This is the maximum fall of a price throuh the time."""
        df = self.portfolio
        drawdowns = 100 * (df.base/df.base.cummax()-1)
        max_drawdown = min(drawdowns) # weird but thats it : we are looking at the maximum negative drawdown.
        return max_drawdown

    def ulcer_index(self):
        """Compute the Ulcer index. This the continuous way of Max Drawdown. Instead of looking at the worst fall, ulcer look through the time
        at the worsts fall and then average them."""
        df = self.portfolio
        drawdowns = 100 * (df.base/pd.Series(df.base).cummax()-1)
        drawdowns_transf = drawdowns.map(lambda x: x**2)
        ulcer = np.sqrt(drawdowns_transf.mean())
        return ulcer

    def annualized_vol(self, freq='D'):
        """Compute the annualized volatility of a portfolio.
        :param freq: Frequency of the returns : 'D' or 'M' for respectively Daily returns or Monthly returns."""
        if(freq=='D'):
            daily_return = self.extract_daily_return()
            return daily_return.std() * np.sqrt(252) * 100
        elif(freq=='M'):
            monthly_return = self.extract_monthly_return()
            return monthly_return.std() * np.sqrt(12) * 100
        else:
            raise KeyError('''Non valid frequency argument. Should be 'D' or 'M' for respectively Daily or Monthly Volatility ''')

    def annualized_return(self):
        """Compute the annualized return of a portfolio. For each year, we compute the annual rate of return."""
        df = self.portfolio
        n_year = len(pd.date_range(df.index[0], df.index[-1], freq='B')) / 252
        final_value = df.base.values[-1]
        ann_return = (final_value**(1/n_year)-1) * 100
        return ann_return

    def sharpe_ratio(self, freq='D', risk_free_rate=False):
        """Compute the Sharpe Ratio of a portfolio.
        :param freq: Frequency of the returns : 'D' or 'M' for respectively Daily returns or Monthly returns.
        :param risk_free_rate: Boolean (True or False for respectively considering or not considering the current risk free rate)."""
        if(risk_free_rate):
            r = quandl.get("USTREASURY/YIELD")['3 MO'].iloc[-1]
        else:
            r = 0
        if(freq=='D'):
            daily_return = self.extract_daily_return()
            return_portfolio = daily_return
            annualized_coef = 252
        elif(freq=='M'):
            monthly_return = self.extract_monthly_return()
            return_portfolio = monthly_return
            annualized_coef = 12
        else:
            raise KeyError('''Non valid frequency argument. Should be 'D' or 'M' for respectively Daily or Monthly Sharpe Ratio ''')
        sharpe =  (return_portfolio.mean()*annualized_coef - r) / (return_portfolio.std() * np.sqrt(annualized_coef))
        return sharpe

    def performance_statistics(self, risk_free_rate=False):
        """Compute a statistic table of a portfolio.
        :param risk_free_rate: Boolean (True or False for respectively considering or not considering the current risk free rate)."""
        statistics = pd.DataFrame(index=[0])
        statistics['Max DD'] = self.max_drawdown()
        statistics['Max DD'] = "{}%".format(statistics['Max DD'].round(2).values[0])
        statistics['DSharpe'] = self.sharpe_ratio('D', risk_free_rate)
        statistics['DSharpe'] = statistics['DSharpe'].round(2).values
        statistics['MSharpe'] = self.sharpe_ratio('M', risk_free_rate)
        statistics['MSharpe'] = statistics['MSharpe'].round(2).values
        statistics['Dvol'] = self.annualized_vol('D')
        statistics['Dvol'] = "{}%".format(statistics['Dvol'].round(2).values[0])
        statistics['Mvol'] = self.annualized_vol('M')
        statistics['Mvol'] = "{}%".format(statistics['Mvol'].round(2).values[0])
        statistics['Ann. Return'] = self.annualized_return()
        statistics['Ann. Return'] = "{}%".format(statistics['Ann. Return'].round(2).values[0])
        statistics = statistics[['Ann. Return', 'Max DD', 'DSharpe', 'MSharpe', 'Dvol', 'Mvol']]
        return statistics

    def monthly_return(self):
        """Compute a nice table of the Monthly returns of a portfolio."""
        df = self.portfolio
        monthly_return = pd.DataFrame(self.extract_monthly_return())
        monthly_return['month'] = monthly_return.index.map(lambda x: x.month).values
        monthly_return['year'] = monthly_return.index.map(lambda x: x.year).values
        monthly_return = monthly_return.set_index('year')
        month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Agu', 'Sep', 'Oct', 'Nov', 'Dec']
        df_monthly_return = pd.DataFrame(index=set(monthly_return.index), columns=month_name)
        for i,month in enumerate(month_name):
            df_monthly_return[month] = 100 * monthly_return[(monthly_return['month']==i+1)]['daily_return']
        df_monthly_return['Year'] = df['daily_return'].groupby([df.index.map(lambda x: x.year)]).apply(lambda x: round(self.__geometric_return(x)*100,2))
        df_monthly_return = df_monthly_return.round(2)
        df_monthly_return = df_monthly_return.sort_index()
        df_monthly_return = df_monthly_return.fillna(0)
        return df_monthly_return

    def plot_performance(self):
        """Plot the performance of a portfolio."""
        df = self.portfolio

        plt.figure(figsize=(15,5))
        plt.plot(df.base, color='black', linewidth = 0.75)
        plt.title('Performance')
        plt.show()

        plt.figure(figsize=(15,2))
        plt.plot(df['daily_return'], color='black', linewidth = 0.75)
        plt.title('Performance')
        plt.show()

        drawdowns = df.base/df.base.cummax() - 1
        plt.figure(figsize=(15,2))
        plt.plot(drawdowns, color='black', linewidth = 0.75)
        plt.title('Drawdown')
        plt.show()

    def plot_rolling_perf(self):
        """Plot the annual rate of return through the time : for each date, compute the geometric return of the last year returns."""
        plt.figure(figsize=(15,3))
        dr = self.extract_daily_return()
        plt.plot(dr.rolling(252).apply(self.__geometric_return), color='black', linewidth = 0.75)
        plt.axhline(y=0, color='lightgrey', linestyle='-')
        plt.title('Rolling 12 month Performance')
        plt.show()

    def plot_pnl_hist(self):
        """Plot the daily, monthly, quarterly and yearly histogram of returns."""
        plt.figure(figsize=(15,4))
        plt.subplot(141)
        daily_return = self.extract_daily_return()
        plt.hist(daily_return, color='lightgrey', ec='white', bins=15)
        plt.title('Daily PL histgram')

        plt.subplot(142)
        month = self.extract_monthly_return().values
        plt.hist(month, color='lightgrey', ec='white', bins=15)
        plt.title('Monthly PL histgram')

        plt.subplot(143)
        quarter = daily_return.groupby([daily_return.index.map(lambda x: x.year), daily_return.index.map(lambda x: math.ceil(x.month/3.))]).apply(self.__geometric_return).values
        plt.hist(quarter, color='lightgrey', ec='white', bins=15)
        plt.title('Quarterly PL histgram')

        plt.subplot(144)
        year = daily_return.groupby([daily_return.index.map(lambda x: x.year)]).apply(self.__geometric_return).values
        plt.hist(year, color='lightgrey', ec='white', bins=15)
        plt.title('Yearly PL histogram')
        plt.show()

    def plot_benchmark(self, benchmark):
        """Plot the performance of a portfolio with the spot benchmark.
        :param benchmark: Dataframe benchmark."""
        df = self.portfolio
        plt.figure(figsize=(15,5))
        plt.plot(df.index, df.base, color='b', linewidth = 0.75, label='portfolio')
        benchmark = benchmark.loc[df.index, 'close'].values
        plt.plot(df.index, benchmark/benchmark[0], color='red', linewidth = 0.75, label='benchmark')
        plt.legend()
        plt.show()
        return benchmark

    def plot_delta(self):
        """Plot the delta through the time and the delta histogram."""
        df = self.portfolio
        plt.figure(figsize=(15,5))
        plt.vlines(df['delta'].index, [0], df['delta'].values, color='black', linewidth = 0.75)
        plt.title('Delta')
        plt.xlabel('Date')
        plt.ylabel('Delta')
        plt.show()
        
        plt.figure(figsize=(15,5))
        deltas = df.groupby(df.index).apply(lambda x: x['quantity'].sum()*x['delta'].values[0]*1000000)
        plt.hist(deltas.values, bins=50, color='lightgrey', ec='white')
        plt.title('Delta Histogram')
        plt.xlabel('Delta')
        plt.ylabel('Frequency')
        plt.show()

    def plot_performance_on_regime(self, regimes):
        """ Return the performance of a portfolio based on its Daily returns during 'regimes'.
        On each regime period, we keep the Daily returns of the portfolio during this period and compute the resulted performance.
        :param regimes: List of list of dates indexes."""
        plt.figure(figsize=(15,5))
        for ind,regime in enumerate(regimes):
            daily_return = self.extract_daily_return()
            regime_daily_return = daily_return.loc[daily_return.index.isin(regime)]
            base = (1+regime_daily_return).cumprod()
            plt.plot(base, label='region '+str(ind))
        plt.legend()
        plt.show()

def correlation(portfolios, display=True):
        """Compute the correlation of the strategies of the portfolio based on monthly returns. A plot of the annual correlation (based on monthly returns) can also be displayed.
        :param portfolios: List of Portfolio object.
        :param display: Boolean (True or False for respectively plotting monthly correlations or not)."""
        dfs = [portfolio.portfolio for portfolio in portfolios]
        dfs = intersect_date(dfs)
        for df in dfs:
            portfolio.portfolio = df
        monthly_returns = [portfolio.extract_monthly_return() for portfolio in portfolios]
        df_monthly_returns = pd.DataFrame()
        for ind,mr in enumerate(monthly_returns):
            df_monthly_returns[str(ind)] = mr
        if(display):
            plt.figure(figsize=(15,5))    
            for strat1 in range(len(portfolios)-1):
                for strat2 in range(strat1+1,len(portfolios)):
                    plt.plot(df_monthly_returns[str(strat1)].rolling(12).corr(df_monthly_returns[str(strat2)]).values, label='Strat {} vs {}'.format(strat1, strat2))
                    plt.ylim(-1.05, 1.05)
                    plt.axhline(y=0, color='grey', linestyle='-', alpha=0.05)
                    plt.legend()
            plt.title('1Y monthly return correlation')
            plt.show()
        correlation = df_monthly_returns.corr()
        #correlation = correlation.style.background_gradient(cmap='coolwarm').set_precision(2)
        return correlation

if __name__ == '__main__':
    spx_put = import_data('SP', 'put', '2006-01-01')
    spx_spot = import_data('SP', 'spot', '2006-01-01')
    opt = Strategy('my_strategy')
    eom = entry_dates(spx_put.index[0], spx_put.index[-1], 'BM')
    opt.add_instrument(1, spx_put, eom, 30, -1, 'o', 'Roll', 0.25)
    opt.type_investment(1, 'underlying')
    opt.weights(1,1)

    portfolio = Portfolio(opt)
    portfolio.weights({'my_strategy':1})
    portfolio.fit()
    portfolio.VAR(0.05)
    portfolio.ulcer_index()
    portfolio.performance_statistics()
    portfolio.monthly_return()
    portfolio.plot_performance()
    portfolio.plot_rolling_perf()
    portfolio.plot_pnl_hist()
    portfolio.plot_benchmark(spx_spot)
    portfolio.plot_delta()
    bull = MA_regime(spx_spot.close,10,200,'bull')
    bear = MA_regime(spx_spot.close,10,200,'bear')
    portfolio.plot_performance_on_regime([bull, bear])
    correlation = correlation([portfolio,portfolio])