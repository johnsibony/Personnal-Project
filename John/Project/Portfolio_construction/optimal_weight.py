"""
Find optimal weights allocation of strategies using different optimization theory.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

from util import *
from data import *
from portfolio import *
from regime import MA_regime
import random
from deap import algorithms, tools, creator, base
import cvxopt as optz
from cvxopt import blas, solvers
import scipy.optimize as optimize
solvers.options['show_progress'] = False

####################################################### Markowitz optimization #######################################################
    
def markowitz(returns, bounds):
    """Return the optimal tuple (weights strategies, returns, volatilities) acording to Markowitz's portfolio theory.
    :param returns: List of list of returns for each strategies.
    :param bounds: List of tuple of the bounds of the weights strategies. For example : [(0,1), (0,1), (0.5,1)]."""
    n = len(returns)
    if(len(returns)!=len(bounds)):
        raise TypeError('The number of bounds does notmatch the number of strategy. Should have {} bounds'.format(n))
    returns = np.asmatrix(returns)
    N = 100
    mus = [10**(5.0 * t/N - 2.0) for t in range(N)]
    # Convert to cvxopt matrices.
    S = optz.matrix(np.cov(returns))
    pbar = optz.matrix(np.mean(returns, axis=1))
    # Create constraint matrices.
    sgn_bound = [(np.eye(1,n,k=ind)[0]*-1, np.eye(1,n,k=ind)[0]*1) for ind in range(n)]
    sgn_bound = [list(i) for j in sgn_bound for i in j]
    constraint = [np.sign(ind-0.5)*bound for strategy in bounds for ind,bound in enumerate(strategy)]
    G = optz.matrix(sgn_bound).T
    h = optz.matrix(constraint)
    A = optz.matrix(1.0, (1,n))
    b = optz.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming.
    portfolios = []
    for mu in mus:
        portfolios.append(solvers.qp(mu*S, -pbar, G, h, A, b)['x'])
        try:
            portfolios.append(solvers.qp(mu*S, -pbar, G, h, A, b)['x'])
        except:
            pass
    # CALCULATE RISKS AND RETURNS FOR FRONTIER.
    optimal_returns = [blas.dot(pbar, x) for x in portfolios]
    optimal_risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    # CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE.
    m1 = np.polyfit(optimal_returns, optimal_risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO.
    try:
        optimal_weights = solvers.qp(optz.matrix(x1*S), -pbar, G, h, A, b)['x']
    except:
        optimal_weights = np.nan
    optimal_weights = np.asarray(optimal_weights)
    return optimal_weights, optimal_returns, optimal_risks

def plot_efficient_frontiere(returns, optimal_returns, optimal_risks, bounds, n_portfolios=10000):
    """Plot the efficient frontiere of Markowitz's portfolio, with random couple (return,volatility) portfolio.
    :param returns: List of list of returns for each strategies.
    :param optimal_returns: List of returns for optimal couple (returns,volatilities) portfolios.
    :param optimal_risks: List of volatilities for optimal couple (returns,volatilities) portfolios.
    :param bounds: List of tuple of the bounds of the weights strategies. For example : [(0,1), (0,1), (0.5,1)].
    :param n_portfolios: Number of random portfolio."""
    def rand_weights():
        """Returns list of random weights that sum to 1 with 'bounds' constraints."""
        last_weight = [2]
        while(last_weight[0]<bounds[-1][0] or last_weight[0]>bounds[-1][1]):
            weights = [np.random.uniform(bound[0], bound[1], 1) for bound in bounds[:-1]]
            last_weight = 1-sum(weights)
        weights.append(last_weight)
        return weights
    def random_portfolio():
        """Returns the mean and standard deviation of returns for a random portfolio."""
        p = np.asmatrix(np.mean(returns, axis=1))
        w = np.asmatrix(rand_weights())
        C = np.asmatrix(np.cov(returns))
        mu = w.T * p.T
        sigma = np.sqrt(w.T * C * w) 
        # This recursion reduces outliers to keep plots pretty.
        if sigma > 2:
            return random_portfolio(returns)
        return mu, sigma
    means, stds = np.column_stack([random_portfolio() for _ in range(n_portfolios)])
    plt.figure(figsize=(15,5))
    plt.plot(stds, means, 'o')
    plt.ylabel('mean')
    plt.xlabel('std')
    plt.plot(optimal_risks, optimal_returns, 'y-o')
    plt.show()

####################################################### Genetic Algorithm #######################################################

def genetic_algorithm(portfolio, regime, bounds, n_population=50, n_generation=10, tournsize=3, CXPB=0.5, MUTPB=0.2, indpb=0.05, display=True):
    """ Compute optimal weights strategies on each region of 'regime' using genetic algorithm (see: https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html).
        For each region, find the best weights of the strategies that sum to 1 with 'bounds' constraints.
        Return a list of list : the ith list corresponds to the optimal weights strategies on the ith region of 'regime'. For a list region, the ith item corresponds to the strategy id i+1.
    :param portfolio: Portfolio object.
    :param regime: Serie of regime defining the region id (positive incrementing integers matching the index of lists in 'bounds') for each date indexes.
    :param bounds: List of list of tuple of the bounds of the weights strategies for each region of 'regime'. For example : [[(0,1), (0.5,1)], [(0,1), (0,1)], [(0.6,1), (0.2,0.8)]].
                   for 2 strategies with 3 possible regions. For a list region, The ith items corresponds to the bound of the ith strategy of the 'portfolio'.
    :param n_population: Number of individuals on each generation.
    :param n_generation: Number of generation (number of loop optimization).
    :param tournsize: Number of individuals participating in each tournament : among the 'n_population' individuals, choose 'tournsize' random individuals 
                      and then select the k (k=1?) best individuals of the selected set according to the objective function.
    :param CXPB: Probability of mating two individuals. If this probability is realized, then the two individulas will crossover : an equally random part 
                 of their components (= certain weights strategy if only one region in 'regime', certain region otherwise) will swap each other.
                 For example, if we have defined only region id 0 in 'regime' and we have individual1=[0.1,0.5,0.3,0.1] and individual2=[0.7,0.2,0.05,0.05] then the crossover
                 function could randomly returns the swapped individuals : individual1=[0.1,0.2,0.05,0.1] and individual2=[0.7,0.5,0.3,0.05].
                 For example, if we have defined region id 0,1 in 'regime' and we have individual1=[[0.1,0.5,0.3,0.1], [0.1,0.5,0.3,0.1]] and individual2=[[0.3,0.4,0.1,0.1], [0.2,0.2,0.2,0.4]]
                 then the crossover function could randomly returns the swapped individuals : individual1=[[0.1,0.5,0.3,0.1], [0.2,0.2,0.2,0.4]] and individual2=[[0.3,0.4,0.1,0.1]] [0.1,0.5,0.3,0.1].
    :param MUTPB: Probability an individual will mutate.
    :param indpb: If an individual has been chosen for a mutation (by MUTPB parameter), then each of its component will mutate or not according to 'indpb' probability.
    :param display: Boolean ('True' or 'False' to display optimal weight on each generation).
    """
    def check_validity():
        region_id = sorted(set(regime))
        nb_region = len(region_id)
        if(region_id!=list(range(nb_region))):
            raise KeyError('Regime Serie is not valid. Ids should be positive incrementing integers starting from 0 (ex: for 3 region should define ids 0,1,2).')
        if(len(bounds)!=len(region_id)):
            raise KeyError('Number of bounds does not match number of different region of the Serie regime. Should have {} bounds regime.'.format(nb_region))
        nb_strategy = len(portfolio.get_strategy_ids())
        for bounds_regime in bounds:
            if(not isinstance(bounds_regime,list)):
                raise KeyError('Bounds argument should be a list of list (not a list).')
            if(len(bounds_regime)!=nb_strategy):
                raise KeyError('Number of strategy on the region id {} is invalid. Should have {} bounds strategy.'.format(nb_strategy))

    def objective(weight_strategies):
        """ Objective function to maximize. Here we are maximizing the monthly sharpe ratio regularized (sum of weights close to 1) of the portfolio
        It also allows to change weights of strategies depending on regime condition. This switch would occur only when openning a new contract/trade.
        :param weight_strategies: List of list of weights for each strategies on each regime condition. List number i corrresponds to the weights on the ith region of regime.
                  For example, if at date d we open a new contract and we are on region of regime 5, the 5th list of weights will be considered."""
        regularization = np.mean([(sum(weight_strategy)-1)**2 for weight_strategy in weight_strategies])
        weight_strategies = np.array(weight_strategies).T
        weight_strategies = dict(zip(portfolio.get_strategy_ids(), weight_strategies))
        weights = portfolio.get_weight_regime_format()
        for strategy_id in portfolio.strategies:
            weights[strategy_id] = regime.apply(lambda x: weight_strategies[strategy_id][x])
        portfolio.weights(weights)
        portfolio.fit()
        monthly_return = portfolio.extract_monthly_return().values
        return portfolio.sharpe_ratio() - regularization, 
    def initialize_individual():
        """Returns list of list of random weights (= individual) that sum to 1 with 'bounds' constraints for each region of 'regime'."""
        individual = []
        for bound_region in bounds:
            last_weight = bound_region[-1][1]+1
            while(last_weight<bound_region[-1][0] or last_weight>bound_region[-1][1]):
                weights = [np.random.uniform(bound[0], bound[1], 1)[0] for bound in bound_region[:-1]]
                last_weight = 1-sum(weights)
            weights.append(last_weight)
            individual.append(weights)
        return individual
    def feasible(weight_strategies):
        """Constraint function for individual. Returns True if feasible, False otherwise."""
        for region_id,weight_strategy in enumerate(weight_strategies):
            for ind,weight in enumerate(weight_strategy):
                if(weight<bounds[region_id][ind][0] or weight>bounds[region_id][ind][1]):
                    return False
        return True
    def distance(weight_strategies):
        """Distance function to the constraint region."""
        return - sum([(weight - 0.5)**2 for weight_strategy in weight_strategies for weight in weight_strategy])
    def cxTwoPoint(ind1, ind2):
        """Executes a two-point crossover on the two individuals by swaping each other a random part of their component.
        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover."""
        nb_region = len(bounds)
        if(nb_region==1):
            ind1, ind2 = ind1[0], ind2[0]  
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        if(nb_region==1):
            return [ind1], [ind2]
        else:
            return ind1, ind2
    def mutation(individual, indpb):
        """Mutation of each component of an individual.
        :param individual: List of list of weights strategies (defining an individual) for each region of 'regime'.
        :param indpb: If an individual has been chosen for a mutation (by MUTPB parameter), then each of its component will mutate or not according to 'indpb' probability."""
        for region_id in range(len(bounds)):
            nb_strategies = len(portfolio.strategies)
            mut = np.random.random_sample(nb_strategies)
            ind_mut = np.where((mut<=indpb))[0]
            for strategy in ind_mut:
                individual[region_id][strategy] = np.random.uniform(bounds[region_id][strategy][0], bounds[region_id][strategy][1], 1)[0]
            return individual,
    def fit():
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, initialize_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("r", objective)
        toolbox.register("mate", cxTwoPoint)
        toolbox.register("mutate", mutation, indpb=indpb)
        toolbox.register("select", tools.selTournament, tournsize=tournsize)
        toolbox.decorate("r", tools.DeltaPenality(feasible, 0, distance))
        pop = toolbox.population(n=n_population)
        fitnesses = list(map(toolbox.r, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        fits = [ind.fitness.values[0] for ind in pop]
        for generation in range(n_generation):
            print("Generation {} :".format(generation))
            if(generation==0):
                optimal_weight, optimal_fit = [], max(fits)
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.r, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]
            if(display):
                print("Max objective function: {} \nOptimal weight: {} \n".format(max(fits), pop[fits.index(max(fits))]))
            if(max(fits)>optimal_fit):
                optimal_weight = pop[fits.index(max(fits))]
        return optimal_weight
    check_validity()
    optimal_weight = fit()
    return optimal_weight

if __name__ == '__main__':
    spx_put = import_data('SP', 'put', '2006-01-01')
    spx_future = import_data('SP', 'future', '2006-01-01')
    spx_future['underlying'] = spx_spot['close']
    spx_spot = import_data('SP', 'spot', '2006-01-01')
    eom = entry_dates(spx_put.index[0], spx_put.index[-1], 'BM')
    strat1 = Strategy('strategy1')
    strat1.add_instrument(1, spx_put, eom, 30, -1, 'o', 'Roll', 0.2)
    strat1.add_instrument(2, spx_future, eom, 35, 1, 'f', 'Roll')
    strat1.type_investment(1, 'underlying')
    strat1.type_investment(2, 'underlying')
    strat1.weights(1,1)
    strat1.weights(2,0.2)
    strat2 = Strategy('strategy2')
    strat2.add_instrument(1, spx_put, eom, 30, -1, 'o', 'Roll', 0.25)
    strat2.add_instrument(2, spx_future, eom, 35, 1, 'f', 'Roll')
    strat2.type_investment(1, 'underlying')
    strat2.type_investment(2, 'underlying')
    strat2.weights(1,1)
    strat2.weights(2,0.25)
    strat3 = Strategy('strategy3')
    strat3.add_instrument(1, spx_put, eom, 30, -1, 'o', 'Roll', 0.35)
    strat3.add_instrument(2, spx_future, eom, 35, 1, 'f', 'Roll')
    strat3.type_investment(1, 'underlying')
    strat3.type_investment(2, 'underlying')
    strat3.weights(1,1)
    strat3.weights(2,0.35)
    strat4 = Strategy('strategy4')
    strat4.add_instrument(1, spx_put, eom, 30, -1, 'o', 'Roll', 0.55)
    strat4.add_instrument(2, spx_future, eom, 35, 1, 'f', 'Roll')
    strat4.type_investment(1, 'underlying')
    strat4.type_investment(2, 'underlying')
    strat4.weights(1,1)
    strat4.weights(2,0.55)
    
    ####################################################### Markowitz optimization #######################################################
    portfolio1, portfolio2, portfolio3, portfolio4 = Portfolio(strat1), Portfolio(strat2), Portfolio(strat3), Portfolio(strat4)
    portfolio1.weights({'strategy1':1}), portfolio2.weights({'strategy2':1}), portfolio3.weights({'strategy3':1}), portfolio4.weights({'strategy4':1})
    portfolio1.fit(), portfolio2.fit(), portfolio3.fit(), portfolio4.fit()
    return1,return2,return3,returns4 = portfolio1.extract_monthly_return(), portfolio2.extract_monthly_return(), portfolio3.extract_monthly_return(), portfolio4.extract_monthly_return()
    return1,return2,return3,returns4 = intersect_date([return1,return2,return3,returns4])
    returns = np.array([return1,return2,return3,returns4])
    bounds = [(0.25,1), (0.1,1), (0.1,0.25), (0.1,0.25)]
    optimal_weights, optimal_returns, optimal_risks = markowitz(returns,bounds)
    plot_efficient_frontiere(returns, optimal_returns, optimal_risks, bounds)

    ####################################################### Genetic Algorithm #######################################################
    portfolio = Portfolio(strat1, strat2, strat3, strat4)
    portfolio.weights({'strategy1':1, 'strategy2':1, 'strategy3':1, 'strategy4':1})
    bounds = [[(0.25,1), (0.1,1), (0.1,0.25), (0.1,0.25)], [(0.25,1), (0.1,1), (0.1,0.25), (0.1,0.25)]]
    bear = MA_regime(spx_spot.close, 10, 200, 'bear')
    bull = MA_regime(spx_spot.close, 10, 200, 'bull')
    regime = pd.Series(0, index=set(spx_put.index).union(bear,bull))
    regime.loc[bull] = 1
    genetic_algorithm(portfolio, regime, bounds, n_population=50, n_generation=10, tournsize=3, CXPB=0.5, MUTPB=0.2, indpb=0.05, display=True)