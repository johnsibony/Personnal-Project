Project Architecture
--------------------

-   The folder 'Portfolio\_construction' contains files to build
    portfolios. The remaining folder depend on it to build portfolios
    based on different strategies.

-   The folder 'Dash' contains files to create a GUI using Dash for
    Python.

-   The folders 'Hedge\_premia', 'Tail' and 'Term\_structure' build
    portfolios of respectively New Hedge Premia strategies, Tail Hedging
    strategies and Arbitrage on Term Structure strategies. Each folder
    contains a main.py file containing the function that build their
    portfolio strategies. The remaning files are tool/analysis function
    used for the main.py file.

-   The folder 'Lyft' contains a jupyter notebook file for the analysis
    of Lyft's IPO and a ppt presentation.

-   Each files contain test functions. By executing a file, the tests
    will be executed.

Portfolio construction
----------------------

##### How a portfolio is defined ?

A portfolio is a set of strategies. A strategy is a set of instruments.
An instrument is a set of contracts/trades defined by several features.

##### How an instrument is defined ? 
Here is the instrument's arguments: 
- id: Id of the instrument (typicaly an integer). 
- data: The Dataframe containing the contract features (price, delta, date,
maturity...). 
- entry: The date to open a contract. 
- dte: The minimum number of days of the life of the contracts. 
- position: 1 for long, -1 for short. 
- tp: 'f' for future, 'o' for option. 
- end: 'Roll' to roll the contracts opened at entry dates, 'Expiry' to let to expiry. 
- weight: Weight to apply to compute quantity of contracts to open (see the following argument). 
- investment: If 'underlying', quantity of contract to open at entry dates will be:
weight\*value\_portfolio/underlying\_price. If 'value':
weight\*value\_portfolio/premium\_contract\_to\_open. 
- delta (Optional): Target delta of the option contracts to open at entry dates.
- strike (Optional): Target strike of the option contracts to open at
entry dates.

Here is the instrument's methods to adjust the contracts: 
- keep\_trade(dates): Keep contracts for which open date belongs to 'dates'. - stop\_date(dates): Close contracts at given 'dates'. 
- stop\_gain(n\_time): Stop trades when the value is 'n\_time' times the entry value. 
- pause(dates, time): Close all the contract openned between each dates of 'dates' and 'time' days after.

##### Example:

A portfolio on S&P and VIX index is composed of 2 strategies (S1 and
S2).

S1 is composed of two isntruments I1 and I2. I1 is short put S&P 25d
rolled every month with 1 weight and closing the contract on bull
market. I2 is a short future S&P with 0.25 weight (for delta hedging).

S2 is composed of two isntruments I1 and I2. I1 is long call VIX 5d
rolled every month with 1 weight and closing the contract on bear
market. I2 is a short future S&VIX with 0.05 weight (for delta hedging).

Start Guide (for developers)
----------------------------

By importing files on the folder 'Portfolio\_construction', one can
create its own portfolio. The following code reproduce the portfolio
stated above on the Exemple section

Import:

    import sys
    sys.path.append('Portfolio_construction') #if the current path is at the same level as the folder 'Portfolio_construction'
    from data import *
    from portfolio import *
    from regime import MA_regime

Data:

    spx_spot = import_data('SP', 'spot', '2006-01-01')
    spx_put = import_data('SP', 'put', '2006-01-01')
    spx_fut = import_data('SP', 'future', '2006-01-01')
    spx_fut['underlying'] = spx_spot.close

    vix_spot = import_data('VIX', 'spot', '2006-01-01')
    vix_call = import_data('VIX', 'call', '2006-01-01')
    vix_fut = import_data('VIX', 'future', '2006-01-01')
    vix_fut['underlying'] = vix_spot.close

    bull = MA_regime(spx_spot.close,10,200,'bull')
    bear = MA_regime(vix_spot.close,10,200,'bear')

Strategies:

    S1 = Strategy(name='S1')
    eom_put = entry_dates(spx_put.index[0], spx_put.index[-1], 'BM')
    eom_fut = entry_dates(spx_fut.index[0], spx_fut.index[-1], 'BM')
    S1.add_instrument(instrument_id=1, spx_put, eom_put, 30, -1, 'o', 'Roll', 0.25)
    S1.add_instrument(instrument_id=2, spx_fut, eom_fut, 30, -1, 'f', 'Roll')
    S1.adjust_instrument(1, 'stop_date', bull) #first argument is the id of the instrument.
    S1.type_investment(1, 'underlying') #first argument is the id of the instrument.
    S1.type_investment(2, 'underlying')
    S1.weights(1,1) #first argument is the id of the instrument.
    S1.weights(2,0.25)

    S2 = Strategy(name='S2')
    eom_call = entry_dates(vix_call.index[0], vix_call.index[-1], 'BM')
    eom_fut = entry_dates(vix_fut.index[0], vix_fut.index[-1], 'BM')
    S2.add_instrument(instrument_id=1, vix_call, eom_call, 30, 1, 'o', 'Roll', 0.05)
    S2.adjust_instrument(1, 'stop_date', bear)
    S2.add_instrument(instrument_id=2, vix_fut, eom_fut, 30, -1, 'f', 'Roll')
    S2.type_investment(1, 'underlying')
    S2.type_investment(2, 'underlying')
    S2.weights(1,1)
    S2.weights(2,0.05)

Portfolio:

    portfolio = Portfolio(S1, S2)
    portfolio.weights({'S1':1, 'S2':1})
    portfolio.fit()
    # Display/return informations on the portfolio:
    portfolio.get_strategy_ids()
    portfolio.strategies[1].instruments[2] #Dataframe of the trades of Strategy 1 - Instrument 2 (Short future S&P).
    portfolio.portfolio #Dataframe of Portfolio value.
    portfolio.VAR(0.05)
    portfolio.ulcer_index()
    portfolio.performance_statistics()
    portfolio.monthly_return()
    portfolio.plot_performance()
    portfolio.plot_rolling_perf()
    portfolio.plot_pnl_hist()
    portfolio.plot_benchmark(spx_spot)
    portfolio.plot_delta()

Remarks: 
- The Strategy class contain a method 'delete\_instrument' to delete an instrument in the strategy. 
- Weights can be defined for instrument but also for strategy. If 2 strategies has weights of 0.2 and 0.8. If the first strategy has 2 instruments with a weight of 0.1 and 0.3. If the second strategy has 1 instruments with a weight of 0.1. Instruments of the first strategy will then have a weight of 0.2\*0.1 and 0.2\*0.3 and instrument of the second strategy will have a weight of 0.8\*0.1. 
- Defining weight by strategy allow us to use weight optimization algorithm (find the best weight allocation between strategies that sum to 1). A file optimal\_weight.py in the folder 'Portfolio\_construction' contains 2 different functions to compute optimal weights : by using Markowitz's theory or by maximizing the daily Sharpe Ratio with Genetic algorithm. 
- The method 'weights' of the class Strategy and Portfolio accept Dataframe to change weights depending on dates. Call the method 'get\_weight\_regime\_format(id\_instrument)' of the class Strategy to get the Dataframe weight to fill. Call the method 'get\_weight\_regime\_format' of the class Portfolio to get the Dataframe weight to fill.

Start Guide (for users)
-----------------------

Go to the folder 'Portfolio\_construction\Dash' and run the file
portfolio.py. A link will be returned to get access to the platform of
backtesting.

Further Improvement
-------------------

-   Only Future and Option contract are allowed to define a portfolio.
    User should also be able being long/short on any asset (equity
    asset...).
-   User can define weights of a portfolio depending on time (not
    necessarly constant weight through the time) which are used to
    compute quantity of contract to purshase. Thus, only weight defined
    on dates corresponding to openning a contract will be used. For
    example, if we defined a weight of 0.5 on '2006-01-01' and 0.5 on
    '2006-05-05' - and we open contracts only at the beginning of each
    year - only the 0.5 weight on '2006-01-01' will be taking into
    account. However a user should also have the possibility to change
    the weight of the holding contract (buy/sell more contracts than you
    actually have). For example, if I oppened a contract on '2006-01-01'
    that expires on '2006-05-05' with 0.5 weight which gave me a
    quantity of 100 contracts to open, I could have the possibility to
    increase/decrease the quantity on any date during the lifetime of
    the contract.

