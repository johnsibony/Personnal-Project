"""
Statistics of HedgePremia for presentation.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

import sys
sys.path.append('../Portfolio_construction')
from data import *
from main import *
import matplotlib.pyplot as plt
import numpy as np

def alpha_contribution(delta_neutral, trend_adj, dstat_adj, tail, ty, benchmark):
    """Plot the Yearly performance of the strategies : Neutral+Trend+Dstat (compared to 'benchmark'), TY, Tail, HedgePremia.
    :param delta_neutral: Delta neutral strategy of HedgePremia.
    :param trend_adj: Trend adjustment strategy of HedgePremia.
    :param dstat_adj: Dstat adjustment strategy of HedgePremia.
    :param tail: Tail hedging strategy of HedgePremia.
    :param ty: TY strategy of HedgePremia.
    :param benchmark: Benchmark strategy of HedgePremia."""
    def fill_yearly_return(returns, years):
        """Some strategy do not have trade (and so no yearly returns) for certain year which produce a gap year. We create a 0 return for these years.
        :param returns: DataFrame of the yearly return of a portfolio.
        :param years: List of the full years in Datetime format."""
        yearly_return = pd.DataFrame(index=years, columns=['return'])
        yearly_return['return'] = returns
        yearly_return = yearly_return.fillna(0)
        yearly_return = yearly_return.values
        yearly_return = [y_return[0] for y_return in yearly_return]
        return yearly_return

    portfolio1 = Portfolio(delta_neutral, trend_adj, dstat_adj)
    portfolio1.weights({'Delta neutral':1, 'Trend adjustment':1, 'DSTAT adjustment':1})
    portfolio1.fit()
    portfolio2 = Portfolio(ty)
    portfolio2.weights({'TY':1})
    portfolio2.fit()
    portfolio3 = Portfolio(tail)
    portfolio3.weights({'Tail':1})
    portfolio3.fit()
    portfolio4 = Portfolio(delta_neutral, trend_adj, dstat_adj, ty, tail)
    portfolio4.weights({'Delta neutral':1, 'Trend adjustment':1,'DSTAT adjustment':1, 'TY':1, 'Tail':1})
    portfolio4.fit()
    benchmark = Portfolio(benchmark)
    benchmark.weights({'Benchmark':1})
    benchmark.fit()

    labels = portfolio1.monthly_return()['Year'].index
    s1 = portfolio1.monthly_return()['Year'].values-benchmark.monthly_return()['Year'].values
    s2 = portfolio2.monthly_return()['Year']
    s3 = portfolio2.monthly_return()['Year']
    s3 = fill_yearly_return(s3, labels)
    s4 = portfolio4.monthly_return()['Year']
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots()
    ax.bar(x - width, s1, width, label='Neutral+Trend+Dstat')
    ax.bar(x - width/2, s2, width, label='TY')
    ax.bar(x + width/2, s3, width, label='Tail')
    ax.bar(x + width, s4, width, label='HP')
    ax.set_ylabel('Pnl')
    ax.set_title('Alpha')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    #fig.savefig('alpha_contribution')
    plt.show()

if __name__ == '__main__':
    spx_spot = import_data('SP', 'spot', '1975-01-01')
    spx_call = import_data('SP', 'call', '2006-01-01')
    spx_put = import_data('SP', 'put', '2006-01-01', freq=('EW3', 'ES'))
    spx_future = import_data('SP', 'future', '2006-01-01')
    spx_future['underlying'] = spx_spot['close']
    bond_future = import_data('10Ybond', 'future', '2006-01-01')
    delta_neutre = delta_neutral(spx_put, spx_future, spx_spot)
    trend_adj = trend_adjustment(spx_future, spx_spot)
    dstat_adj = dstat_adjustment(spx_future, spx_spot)
    tail = tail_hedging(spx_put, spx_spot)
    ty = TY(bond_future, spx_spot)
    bench = benchmark(spx_future)
    alpha_contribution(delta_neutre, trend_adj, dstat_adj, tail, ty, bench)