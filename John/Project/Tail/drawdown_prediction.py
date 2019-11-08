"""
Prediction of Drawdowns using interpretable Machine Learning algorithm (Decision Tree).
"""

# Author: John Sibony <john.sibony@hotmail.fr>

import sys
sys.path.append('../Portfolio_construction')
from portfolio import *
from regime import DSTAT_regime, DSTAT
from tail_event import *
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image 
import pydot
from sklearn import preprocessing

def accuracy(y_true, y_pred):
    """ Display confusion matrix with a pie chart of precision and recall. 
    :param y_true: List of true values (0: No Drawdown, 1: Drawdown).
    :param y_pred: List of predicted values (0: No Drawdown, 1: Drawdown)."""
    cm = confusion_matrix(y_true, y_pred)
    classes = np.array(['No Drawdown', 'Drawdown'])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, ylabel='True', xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15,5))
    recall, precision = recall_score(y_true, y_pred), precision_score(y_true, y_pred)
    plt.subplot(121)
    labels = 'True positives', 'False negatives'
    sizes = [recall, 1-recall]
    colors = ['lawngreen', 'red']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=130)
    plt.axis('equal')
    plt.title('% of true Drawdown detected')
    plt.subplot(122)
    labels = 'True positives', 'False positves'
    sizes = [precision, 1-precision]
    colors = ['lawngreen', 'red']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=150)
    plt.axis('equal')
    plt.title('% of accurate prediction')
    plt.show()

def gap_prediction(true_period, pred_date):
    """ Gap business days between DSTAT prediction and true DD. 
        For each date of true drawdown, the closest previous predicted date will be considered to compute gap days. But if the previous closest date as already been 
        associated with another true drawdown, the true drawdown date won't be considered. This is the case when there is no prediction date between n>=2 true drawdowns.
        :param true_period: List of tuple (Start date, Min date peak) representing drawdowns period.
        :param pred_date: List of predicted dates in datetime format."""
    pred_date = np.array(pred_date)
    plt.figure(figsize=(25,5))
    plt.subplot(121)
    plt.axvline(pred_date[0], color='red', label='Prediction')
    [plt.axvline(x, color='red') for x in pred_date[1:]]
    plt.axvspan(true_period[0][0], true_period[0][1], color='grey', alpha=0.5, label='Drawdown period')
    signal, gap = -1, {}
    for drawdown_period in true_period[1:]:
        start, end = drawdown_period[0], drawdown_period[1]
        plt.axvspan(start, end, color='grey', alpha=0.5)
        days, ind = min(((gap.days,ind) for ind,gap in enumerate(start-pred_date) if gap.days>=0))
        if(ind==signal):
            continue
        signal = ind
        gap[start] = days
    plt.legend(loc='upper right')
    plt.title('DSTAT predicition and true Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Triggered Drawdown')
    plt.subplot(122)
    plt.scatter(gap.keys(), gap.values(), color='dodgerblue')
    plt.xlabel('DD date')
    plt.ylabel('Gap prediciton (Business days)')
    plt.title('Gap days between true and closest predicted drawdown')
    plt.show()
    
class DSTAT_predictor():
    """Predict Drawdowns dates using best DSTAT parameters.
    DSTAT = (MA(ma1)-MA(ma2)) / (MA(ma2)*VOL(rv))."""
    
    def __init__(self, df_drawdown, underlying_dstat, trend_dstat, quantile_dstat):
        """ 
        :param df_drawdown: Drawdown Dataframe returned by 'tail_by_drawdown' function contaning true period of drawdowns.
        :param underlying_dstat: Dataframe of spot price to compute DSTAT indicator (generally SPX or VIX).
        :param trend_dstat: Integer of the DSTAT trend (1, or -1 for respecively extreme low or extreme high DSTAT values of 'df_dstat').
                            Depending on the 'underlying_dstat' data, high or low value of dstat could predict drawdowns (ex: high dstat values for SP underlying can be a good predictor of negative drawdowns).
        :param quantile_dstat: Quantile of the DSTAT to quantify extreme values defined by 'trend_dstat'.
                               (normally, if 'trend_dstat'==1, 'quantile_dstat' should be low. If 'trend_dstat'==-1, 'quantile_dstat' should be high)."""
        self.ma1, self.ma2, self.rv, self.roll_year = 0, 0, 0, 0
        self.df_drawdown = df_drawdown
        self.underlying_dstat = underlying_dstat
        self.trend_dstat = trend_dstat
        if(trend_dstat==-1): #high value of dstat can predict drawdowns.
            self.low_qt = 0.05 #thus, we do not care about the low dstat quantile.
            self.high_qt = quantile_dstat
        elif(trend_dstat==1): #low value of dstat can predict drawdowns.
            self.high_qt = 0.95 #thus, we do not care about the high dstat quantile.
            self.low_qt = quantile_dstat
        else:
            raise KeyError("""Trend argument is invalid. Should be -1, or 1 for respectively extreme high or extreme low values of DSTAT. """)
    
    def fit(self, metric):
        """ Optimize the parameters 'ma1', 'ma2', 'rv' and 'roll_year' of the DSTAT indicator according to 'metric'.
        :param metric: Metric to optimize (should be 'precision', or 'recall' or 'f1'. See the module sklearn.metrics for further information)."""
        print('---------- Finding best parameters ----------')
        print('Estimation : 5min')
        if(metric=='f1'):
            metric = f1_score
        elif(metric=='precision'):
            metric = precision_score
        elif(metric=='recall'):
            metric = recall_score
        else:
            raise KeyError("""Metric argument is invalid. Should be 'precision', or 'recall' or 'f1'. """)
        true_dates = self.df_drawdown['Start'].values
        y_true = pd.Series(0, index=set(self.underlying_dstat.index))
        y_true.loc[true_dates] = 1
        ma1_values, ma2_values, rv_values, roll_year_values = range(1,6), range(6, 100), range(21, 127, 21), range(1, 6)
        scores = np.zeros((len(ma1_values), len(ma2_values), len(rv_values), len(roll_year_values)))
        for ind1, ma1 in enumerate(ma1_values):
            for ind2,ma2 in enumerate(ma2_values):
                for ind3,rv in enumerate(rv_values):
                    for ind4,roll_year in enumerate(roll_year_values):
                        pred_dates = self.prediction(ma1, ma2, rv, roll_year)
                        y_pred = pd.Series(0, index=set(self.underlying_dstat.index))
                        y_pred.loc[pred_dates] = 1
                        scores[ind1][ind2][ind3][ind4] = metric(y_true, y_pred)
        ind1, ind2, ind3, ind4 = np.where(scores==scores.max())
        self.ma1, self.ma2, self.rv, self.roll_year = ma1_values[ind1[0]], ma2_values[ind2[0]], rv_values[ind3[0]], roll_year_values[ind4[0]]
        
    def prediction(self, ma1, ma2, rv, roll_year):
        """ Predict dates of drawdown.
        :param ma1: Number of business days for the MA1 indicator. 
        :param ma2: Number of business days for the MA2 indicator. 
        :param rv: Number of business days for the ROLLING_VOL indicator. 
        :param roll_year: Number of rolling year to compute extreme DSTAT values on each date."""
        dates = DSTAT_regime(self.underlying_dstat, ma1, ma2, rv, roll_year, self.low_qt, self.high_qt, self.trend_dstat)
        return dates
    
    def performance(self):
        """ Display confusion matrix with a pie chart and gap business days between DSTAT prediction and true DD. """
        true_dates = self.df_drawdown['Start'].values
        y_true = pd.Series(0, index=set(self.underlying_dstat.index))
        y_true.loc[true_dates] = 1
        pred_dates = self.prediction(self.ma1, self.ma2, self.rv, self.roll_year)
        y_pred = pd.Series(0, index=set(self.underlying_dstat.index))
        y_pred.loc[pred_dates] = 1
        accuracy(y_true.values, y_pred.values)
        true_period = list(zip(self.df_drawdown['Start'],self.df_drawdown['Min Date']))
        gap_prediction(true_period, pred_dates)

    def best_parameters(self):
        """ Return a tuple of the DSTAT optimal parameters ma1, ma2, rv and roll_year. 
        DSTAT = (MA(ma1)-MA(ma2)) / (MA(ma2)*VOL(rv))."""
        return self.ma1, self.ma2, self.rv, self.roll_year

class Decision_tree_predictor:
    """Predict Drawdowns dates based on DSTAT values, MA and SP return using Decision tree classifier algorithm. Best parameters of the DSTAT can be computed using the previous class.
    A trainset will be used to train the algorithm. A valset will be used to find the best parameters of algorithms run in the trainset. A testset will be used to display performance of the best algorithm choosen."""
    
    def __init__(self, df_drawdown, dstat, spx_spot, train_size=0.85):
        """ 
        :param df_drawdown: Drawdown Dataframe returned by 'tail_by_drawdown' function contaning true period of drawdowns.
        :param dstat: Serie of dstat values.
        :param spx_spot: SP spot prices.
        :param train_size: Percentage of true drawdowns to keep on the trainset (see the method __split_dataset below)."""
        dates = sorted(set(spx_spot.index).intersection(dstat.index))
        dstat, spx_spot = dstat.loc[dates], spx_spot.loc[dates]
        X = pd.DataFrame(index=dates, columns=['dstat', 'sp_return', 'MA [-5,0]', 'MA [-10,0]', 'MA [-20,-10]', 'MA [-30,-10]', 'MA [-60,-10]'])
        X['dstat'], X['sp_return'], X['MA [-5,0]'], X['MA [-10,0]'], X['MA [-20,-10]'], X['MA [-30,-10]'], X['MA [-60,-10]'] = dstat, spx_spot.daily_return, spx_spot.close.rolling(6).mean(), spx_spot.close.rolling(11).mean(), spx_spot.close.shift(10).rolling(11).mean(), spx_spot.close.shift(10).rolling(21).mean(), spx_spot.close.shift(10).rolling(51).mean()
        X = X.dropna()
        indexes, columns = X.index, X.columns
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X, index=indexes, columns=columns)
        Y = pd.Series(0, index=X.index)
        true_dates = df_drawdown['Start'].values
        true_dates = sorted(set(true_dates).intersection(X.index))
        Y.loc[true_dates] = 1
        self.df_drawdown = df_drawdown
        self.train_size = train_size
        X, Y, self.X_test, self.Y_test = self.__split_dataset(X, Y, train_size)
        self.X_train, self.Y_train, self.X_val, self.Y_val = self.__split_dataset(X, Y, self.train_size**2)
        self.model = np.nan
 
    def __split_dataset(self, X, Y, train_size):
        """ Returns a trainset, valset and testset. 
        The trainset will be used to train the algorithm. The valset will be used to find the best parameters of algorithms run in the trainset. The testset will be used to display performance of the best algorithm choosen.
        Trainset, valset and testset are build successively : trainset from time T1 to T2, valset from time T2 to T3, trainset from time T3 to T4.
        Let's compute only T2, T3 since T1 and T4 are respectively the first and the last date of our dataset. 
        Let n = total number of drawdowns.
        T2 is the middle of the ('train_size'*'train_size'*n)th drawdowns date and the (('train_size'*'train_size'*n)+1)th drawdowns date.
        T3 is the middle of the ('train_size'*n)th drawdowns date and the (('train_size'*n)+1)th drawdowns date.
        :param X: Dataframe of input.
        :param Y: Dataframe of output.
        :param test_size: Percentage of true drawdowns to keep on the trainset."""
        n = len(self.df_drawdown)
        nb_dd_trainset = int(train_size*n)
        last_dd_trainset = self.df_drawdown.iloc[nb_dd_trainset]['Start']
        first_dd_testset = self.df_drawdown.iloc[nb_dd_trainset+1]['Start']
        frontier_dates = pd.date_range(last_dd_trainset, first_dd_testset, freq='B').date
        frontier_dates = sorted(set(frontier_dates).intersection(X.index))
        middle = int(len(frontier_dates)/2)
        frontier_date = frontier_dates[middle]
        X_train, Y_train, X_test, Y_test = X.loc[:frontier_date], Y.loc[:frontier_date], X.loc[frontier_date:], Y.loc[frontier_date:]
        return X_train, Y_train, X_test, Y_test
    
    @staticmethod
    def __gap_custom(y_true, y_pred):
        """ Custom metric. The returned score is the ratio gap/n_pred. 
        The numerator is the sum of the gap business days between prediction dates and next true drawdown date.
        The denominator is the number dates considered as drawdowns by the algorithm (truly or not)."""
        true_dates = y_true[y_true==1].dropna().index
        pred_dates = y_true[y_pred==1].dropna().index
        signal, gap = -1, []
        for true_date in true_dates:
            days, ind = min(((gap.days,ind) for ind,gap in enumerate(true_date-pred_dates) if gap.days>=0))
            if(ind==signal):
                continue
            signal = ind
            gap.append(days)
        score = sum(gap)/sum(y_pred)
        return score

    def fit(self, metric='gap_custom'):
        """ Find the best decision tree fitting the train set (and not the test set!).
        We train different decision tree onthe trainset and look at the score returned by 'metric' on the validation set. We take the best decision tree that maximize the metric.
        Then the performance of the best decision tree is desplayed on the test set that has never been seen before.
        :param metric: Metric to optimize. Should be 'gap_custom' or 'precision', or 'recall' or 'f1'. See the module sklearn.metrics for further information.
                       The 'gap_custom' metric is defined by the previous method above."""
        print('---------- Finding best parameters ----------')
        if(metric=='f1'):
            print('Estimation : 20sec')
            metric = f1_score
        elif(metric=='precision'):
            print('Estimation : 20sec')
            metric = precision_score
        elif(metric=='recall'):
            print('Estimation : 20sec')
            metric = recall_score
        elif(metric=='gap_custom'):
            print('Estimation : 2min')
            metric = self.__gap_custom
        else:
            raise KeyError("""Metric argument is invalid. Should be 'precision', or 'recall', 'f1' or 'gap_custom'. """)
        max_depths, min_samples_splits, min_samples_leafs = np.linspace(1, 32, 32), np.linspace(0.1, 1, 10), np.linspace(0.1, 0.5, 10)
        scores = np.zeros((len(max_depths), len(min_samples_splits), len(min_samples_leafs)))
        for ind1,max_depth in enumerate(max_depths):
            for ind2,min_samples_split in enumerate(min_samples_splits):
                for ind3,min_samples_leaf in enumerate(min_samples_leafs):
                    model = DecisionTreeClassifier(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_depth=max_depth, class_weight='balanced')
                    model.fit(self.X_train, self.Y_train)
                    y_true, y_pred = self.Y_val, model.predict(self.X_val)
                    scores[ind1][ind2][ind3] = metric(y_true, y_pred)
        ind1, ind2, ind3 = np.where(scores==scores.max())
        max_depth = max_depths[ind1[0]]
        min_samples_split = min_samples_splits[ind2[0]]
        min_samples_leaf = min_samples_leafs[ind3[0]]
        decision_tree = DecisionTreeClassifier(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_depth=max_depth, class_weight='balanced')
        X = pd.concat([self.X_train,self.X_val])
        Y = pd.concat([self.Y_train,self.Y_val])
        decision_tree.fit(X, Y)
        self.model = decision_tree

    def rank_features(self):
        """ Returns a list of tuple (feature's name, % importance) ranking the importance of each features. """
        features = self.X_train.columns
        score = self.model.feature_importances_
        rank = list(zip(features, score))
        return rank

    def export(self):
        """ Create a .pdf file to visualize how the algorithm predict drawdowns."""
        features = self.X_train.columns
        dot_data = tree.export_graphviz(self.model, out_file=None, feature_names=features, filled=True, rounded=True)
        (graph,) = pydot.graph_from_dot_data(dot_data)  
        Image(graph.create_png())
        graph.write_pdf('decision_tree_prediction.pdf')

    def performance(self):
        """ Display confusion matrix with a pie chart and gap business days between DSTAT prediction and true DD.
        This method displays performance only on the testset that has never been used during the training of the algorithm."""
        y_true = self.Y_test.values
        y_pred = self.model.predict(self.X_test)
        accuracy(y_true, y_pred)
        n = len(self.df_drawdown)
        df_drawdown_test = self.df_drawdown.iloc[int(n*self.train_size)+1:]
        true_period = list(zip(df_drawdown_test['Start'],df_drawdown_test['Min Date']))
        pred_dates = self.Y_test.iloc[np.where(y_pred==1)].index
        gap_prediction(true_period, pred_dates)

if __name__ == '__main__':
    spx_spot = import_data('SP', 'spot', '1990-01-01')
    spx_spot = daily_return(spx_spot, 'close')
    df_drawdown = tail_by_drawdown(spx_spot, threshold=0.05)
    dstat = DSTAT_predictor(df_drawdown, spx_spot, -1, 0.90)
    dstat.fit('f1')
    dstat.performance()
    ma1, ma2, rv, roll_year = dstat.best_parameters()
    
    dstat = DSTAT(spx_spot, ma1, ma2, rv, roll_year, 0.0, 0.0).dstat # quantile values have no importance. We are only looking for all values of dstat.
    decision_tree = Decision_tree_predictor(df_drawdown, dstat, spx_spot)
    decision_tree.fit(metric='gap_custom')
    decision_tree.rank_features()
    decision_tree.export()
    decision_tree.performance()