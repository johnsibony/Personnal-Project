Remark
----------------------

- Dash platform won't work since the date file has not been uploaded on github.
The data file named 'data.h5' can be find on the Dropbox at C:\Users\Username\Dropbox (TCG)\TCG\3. Research\John\Project\Portfolio_construction\Dash
- Template files can be find on this folder. These files are the accepted format of excel files to define some features of the portfolio (like entry dates, closing dates, weights by date ...).

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
