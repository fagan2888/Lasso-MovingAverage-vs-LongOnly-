'''

uses 1m,3m,6m and 12m moving average to find the next date return using lasso to take long/short position vs longonly position
gives the sharpe ratio and maxdrawdown
files used
all_asset_class.csv
Libor3M.csv

'''


import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from backtest import Strategy,Portfolio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score

class MovingAverageCrossStrategy(Strategy):

    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.df = self.run_classifier()


    def run_classifier(self):
        ''''#Train-Test split'''
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=.8, random_state=1, shuffle=False)
        y_df = pd.DataFrame(index=y_test.index)
        y_df = pd.concat([y_df, y_test], axis=1)
        X_train = X_train.values;
        X_test = X_test.values;
        y_train = y_train.values;
        y_test = y_test.values

        ''''#scaling'''
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        ''''# apply classifier'''
        lassoReg = Lasso(alpha=.01)
        lassoReg.fit(X_train, y_train)

        y_pred_test = lassoReg.predict(X_test)
        y_pred_train = lassoReg.predict(X_train)

        ''''# calculating mse and r-square'''
        mse = np.mean((y_pred_test - y_test) ** 2)
        score_test = lassoReg.score(X_test, y_test)
        score_train = lassoReg.score(X_train, y_train)

        print('The mse is {:.1f}'.format(mse))
        print('The score is for training data {:.5f} and for testing data {:.5f}'.format(score_train, score_test))

        y_df['predict_val'] = y_pred_test
        y_df[['Close_Price', 'predict_val']] = y_df[['Close_Price', 'predict_val']].shift(1)
        y_df = y_df.pct_change().dropna()

        return y_df


    def generate_signals(self):
        sig = pd.DataFrame(index=self.df.index)
        sig = pd.concat([sig,self.df],axis=1)
        sig['signal'] = np.where(self.df['predict_val']>0,1,-1)
        return sig


class MarketOnClosePortfolio(Portfolio):

    def __init__(self,sig):
        self.sig = sig
        self.positions = self.generate_positions()

    def generate_positions(self):
        pos = pd.DataFrame(index=self.sig.index)
        pos['positions'] = self.sig['signal'].mul(self.sig['Close_Price'].values)
        pos = pd.concat([self.sig['Close_Price'],pos['positions']],axis=1)
        return pos

    def backtest_portfolio(self):
        return (self.positions)


def sharpe_and_dd_calcs(df):
    '''
    :param : send df containing pct_change
    :return: sharpe and drawdown values
    '''
    #sharpe ratio
    vol = np.std(df) * np.sqrt(252) * 100
    cumret = (np.cumprod(1 + df) - 1).iloc[-1] * 100
    sharpe = cumret / vol

    #drawdown
    cum_returns = (1 + returns).cumprod()
    maxdrawdown = np.max(1 - cum_returns.div(cum_returns.cummax())) * 100

    return sharpe,maxdrawdown


if __name__ == "__main__":

    ''''#extract security price'''
    name = 'usdBBB' #replace with any of these to get graphs 'crude', 'dollar', 'gold', 'spx', 'emcorp', 'emeq', 'isTSY', 'usdcorp', 'usdBB’, ’usdBBB'
    df = pd.read_csv('all_asset_class.csv', index_col='Date', parse_dates=True)[[name]].dropna()
    df.columns = ['Close_Price']

    ''''#generate moving averages'''
    df['30ma'] = df.Close_Price.rolling(30).mean()
    df['90ma'] = df.Close_Price.rolling(90).mean()
    df['180ma'] = df.Close_Price.rolling(180).mean()
    df['360ma'] = df.Close_Price.rolling(360).mean()
    df.dropna(inplace=True)
    df['pct_change'] = df['Close_Price'].pct_change()
    df['pct_change'] = df['pct_change'].shift(-1)
    df.dropna(inplace=True)

    X = df.drop(['pct_change','Close_Price'], axis=1)
    y = df[['Close_Price']].shift(-1).dropna()
    X = X.iloc[:-1, :]

    mac = MovingAverageCrossStrategy(X,y)
    signals = mac.generate_signals()
    portfolio = MarketOnClosePortfolio(signals)
    returns = portfolio.backtest_portfolio()

    #get sharpe and drawdown
    #extract libor rates
    lib = pd.read_csv('Libor3M.csv', index_col='DATE', parse_dates=True)
    libind = lib.reindex(returns.index)
    libind = pd.DataFrame(pd.to_numeric(libind.USD3MTD156N, errors='coerce')).fillna(method='ffill')
    libind.columns = ['annual']
    libind['daily'] = ((1 + libind['annual']) ** (1 / 252)) - 1
    libind /= 100

    returns.columns = ['rtn_LO ' + name, 'rtn_ML ' + name]
    returns['rtn_LO ' + name] -= libind.daily
    returns['rtn_ML ' + name] -= libind.daily

    sharpe,maxdd = sharpe_and_dd_calcs(returns)
    print('\nSharpe ratios\n{}'.format(sharpe))
    print('Max drawdown in %\n{}'.format(maxdd))

    combo = returns.dropna()
    combo['cumrtn_ML ' + name] = np.cumprod(1+ combo['rtn_ML ' + name]) - 1
    combo['cumrtn_LO ' + name] = np.cumprod(1 + combo['rtn_LO ' + name]) - 1
    combo[['cumrtn_ML ' + name,'cumrtn_LO ' + name]].plot()

    plt.grid(True)
    plt.ylabel('returns')
    plt.show()