#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 19:39:37 2022

@author: jeremyalexander
"""

#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import talib
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
from sklearn import metrics
from scipy.stats import sem
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
#from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler  
from sklearn.calibration import calibration_curve

from statsmodels.formula.api import logit
import statsmodels.api as sm
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.neural_network import MLPClassifier

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import SGD
#from keras.layers import Flatten
#from keras.layers import Dropout
#from keras.layers.convolutional import Conv1D
#from keras.layers.convolutional import MaxPooling1D
#from keras.layers import LSTM
#from keras.utils import to_categorical
#from tqdm.keras import TqdmCallback

import shap 
from yellowbrick.classifier import ROCAUC


#conda install (package) -c conda-forge


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def data_metrics(ticker, start_date, end_date, interval):
    
    ''' measure metrics over a given timeframe.
        Merge back to 1min data to calculate outcome'''
    
    #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    
    #time intervals:
        # 1min
        # 2min
        # 3min
        # 5min
        # 10min
        # 15min
        # 30min
        # 1hr
        # 2hr
        # 3hr
        # 4hr
        # 1day
        # 1week
        # 1month
    
    # one min data
    #One_min = yf.download(tickers = ticker,
    #                       start = start_date, 
    #                       end = end_date, 
    #                       interval = '1m'
    #                       ).reset_index()
    
    # import desired timestamps 
    data = yf.download(tickers = ticker,
                       start = start_date, 
                       end = end_date, 
                       interval = interval
                       ).reset_index()
    
    # moving average
    data[interval + '_MA_20'] = talib.SMA(data['Adj Close'].values, timeperiod = 20)
    data[interval + '_MA_50'] = talib.SMA(data['Adj Close'].values, timeperiod = 50)
    data[interval + '_MA_100'] = talib.SMA(data['Adj Close'].values, timeperiod = 100)
 
    # rate of change
    data[interval + '_ROC_20'] = talib.ROC(data['Adj Close'].values, timeperiod = 20).round(1)
    data[interval + '_ROC_50'] = talib.ROC(data['Adj Close'].values, timeperiod = 50).round(1)
    data[interval + '_ROC_100'] = talib.ROC(data['Adj Close'].values, timeperiod = 100).round(1)

    # rsi
    data[interval + '_RSI'] = talib.RSI(data['Adj Close'].values, timeperiod = 14)
  
    # overbought/oversold for given timeframe
    data[interval + '_OB_OS'] = np.where(data.iloc[:,-1] <= 30, "OS", 
                                np.where(data.iloc[:,-1] >= 70, "OB", np.NaN      
                                ))
    
    # drop data to be replace with 1min data
    #data = data.drop(['Open','High','Low','Close','Adj Close','Volume'], axis = 1)

    # merge 1 min and given time interval
    #df = One_min.merge(data, how = 'left', on = ['Datetime'])
    
    #print(df)
        
    return data


def trendLabel(df, interval):
    
    ''' Assign label to price action from moving average
       Up if MA > close price. Down if MA < close price'''
    
    
    df['20_Trend'] = np.where(df[interval + '_MA_20'] < df['Adj_Close'], 'D', 'U')
    df['50_Trend'] = np.where(df[interval + '_MA_50'] < df['Adj_Close'], 'D', 'U')
    df['100_Trend'] = np.where(df[interval + '_MA_100'] < df['Adj_Close'], 'D', 'U')
                      
    
    return df


def subPrice_threshold(df, interval):
    
    ''' Once OBOS occurs, return the time and price after threshold is met 
    e.g. 10 pips'''
    
    # index of osobs timepoints
    osob = (df[interval + '_OB_OS'] == 'OS') | (df[interval + '_OB_OS'] == 'OB')

    # next high
    Next_High = (df[osob].reset_index()
               .merge(df[['Datetime', 'Adj_Close']], how = 'cross', suffixes = (None, '_Next_High'))
               .query('(High + 0.001 <= Adj_Close_Next_High) & (Datetime < Datetime_Next_High)')
               .drop_duplicates('Datetime').set_index('index').filter(like = '_High'))

    # next low
    Next_Low = (df[osob].reset_index()
               .merge(df[['Datetime', 'Adj_Close']], how = 'cross', suffixes = (None, '_Next_Low'))
               .query('(Low - 0.001 >= Adj_Close_Next_Low) & (Datetime < Datetime_Next_Low)')
               .drop_duplicates('Datetime').set_index('index').filter(like = '_Low'))
    
    
    df1 = pd.concat([Next_High, Next_Low], axis = 1)
    
    df2 = pd.concat([df, df1], axis = 1)
    
    df2 = df2.replace('nan', np.NaN, regex=True)
    
    #print(df2.dropna(subset = '5m_OB_OS'))
    #print(Next_High)
    #print(Next_Low)
    
    return df2


def subPrice_interval(df, interval):
    
    # index of osobs timepoints
    OSOBS = df[df[interval + '_OB_OS'].isin(['OS','OB'])].index

    # return lowest price and timestamp for given interval
    for osob_timepoint in OSOBS:
        
        start = osob_timepoint + 1
        end = osob_timepoint + 5
        
        # slice dataframes for given timespan
        sliced_df = df[start:end]
        
        # return lowest price within given timespan
        lowest_price = min(sliced_df.loc[start:end, 'Low'])
    
        # May need to refine the logic if there are duplicates in the sub dataframe
        lowest_time = sliced_df.loc[sliced_df['Low'] == lowest_price, 'Datetime'].tolist()[0]
        
        df.loc[osob_timepoint, str(5) + '_Lowest_Price'] = lowest_price
        df.loc[osob_timepoint, str(5) + '_Lowest_Time'] = lowest_time
        
        
    # return highest price and timestamp for given interval
    for osob_timepoint in OSOBS:
        
        start = osob_timepoint + 1
        end = osob_timepoint + 5
        
        # slice dataframes for given timespan
        sliced_df = df[start:end]
        
        # return lowest price within given timespan
        lowest_price = max(sliced_df.loc[start:end, 'High'])
            
        # May need to refine the logic if there are duplicates in the sub dataframe
        lowest_time = sliced_df.loc[sliced_df['High'] == lowest_price, 'Datetime'].tolist()[0]
        
        df.loc[osob_timepoint, str(5) + '_Highest_Price'] = lowest_price
        df.loc[osob_timepoint, str(5) + '_Highest_Time'] = lowest_time      
        
        
    return df
    
    
def priceOutcome(df, interval):
    
    df = df.dropna(subset = interval + '_OB_OS').copy()

    df['Outcome'] = np.where(df['Datetime_Next_High'] < df['Datetime_Next_Low'], 1, 
                    np.where(df['Datetime_Next_High'] > df['Datetime_Next_Low'], -1, 
                    np.where(df['Datetime_Next_High'].notnull() & df['Datetime_Next_Low'].isna(), 1,       
                    np.where(df['Datetime_Next_High'].isna() & df['Datetime_Next_Low'].notnull(), -1, 0)
                    )))
    
    df['Cum_Amount'] = np.where((df[interval + '_OB_OS'] == 'OB') & (df['Outcome'] == 1), -10, 
                       np.where((df[interval + '_OB_OS'] == 'OB') & (df['Outcome'] == -1), 10, 
                       np.where((df[interval + '_OB_OS'] == 'OS') & (df['Outcome'] == 1), 10, 
                       np.where((df[interval + '_OB_OS'] == 'OS') & (df['Outcome'] == -1), -10, 
                                0))))
                    
    
    df = df[['Datetime', interval + '_OB_OS', interval + '_ROC_20', interval + '_ROC_50',interval + '_ROC_100',
             '20_Trend','50_Trend','100_Trend','Outcome','Cum_Amount']]

    
    return df
    
    

# input data. specify timeframe
df = data_metrics('EURUSD=X',  '2022-10-06', '2022-10-07', '5m').rename(columns = {'Adj Close' : 'Adj_Close'})

# assign trend labels for given timeframe. moving average above/below close price
df = trendLabel(df, '5m')

# return price and date for next 10 pips (high/low)
df = subPrice_threshold(df, '5m')

# assign outcome 
df_price = priceOutcome(df, '5m')

df_price = df_price[df_price['Outcome'] != 0]


print(df[0:60])
print(df_price)
print(len(df_price))
print(df_price.groupby('5m_OB_OS')['Cum_Amount'].sum())


df_price = df_price.replace({'5m_OB_OS': {'OS' : -1, 'OB' : 1}})
df_price = df_price.replace({'20_Trend': {'D' : -1, 'U' : 1}})
df_price = df_price.replace({'50_Trend': {'D' : -1, 'U' : 1}})
df_price = df_price.replace({'100_Trend': {'D' : -1, 'U' : 1}})



td = df_price[['5m_OB_OS','5m_ROC_20','5m_ROC_50','5m_ROC_100','20_Trend','50_Trend','100_Trend','Outcome']].copy()

cols = [td.columns[-1]] + [col for col in td if col != td.columns[-1]]
td = td[cols]


# define min max scaler
#scaler = MinMaxScaler()
scaler = StandardScaler()
encoder = LabelEncoder()
    
# features training dataset
X = td[['5m_OB_OS','5m_ROC_20','5m_ROC_50','5m_ROC_100','20_Trend','50_Trend','100_Trend']]
X_labels = X.columns
#X = scaler.fit_transform(X)
#X = X.astype('float32')


# class labels.
y = td['Outcome']
y = encoder.fit_transform(y)

#print(df_shots['Goal_Value'].value_counts())
n_classes = td['Outcome'].nunique()
class_names = td['Outcome'].sort_values().unique()
class_names = ['Price_Lower','Price_Higher']


# Split into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)


def XG():
    
    params = {'max_depth': [1,2,3,4,5,6,10],
              'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
              'n_estimators': [100, 200, 500, 1000],
              'colsample_bytree': [0.1, 0.2, 0.3, 0.6, 0.9, 1.0],
             'eta': [0.1,0.2,0.3,0.4,0.5], 
             'reg_lambda': [0,1,5,10],
             'gamma': [0,1,5],
             'subsample': [0.1, 0.2, 0.3, 0.6, 0.9, 1.0],
              'colsample_bylevel': [0.1, 0.2, 0.3, 0.6, 0.9, 1.0],
             }
    
    # create class model
    #model = XGBClassifier(eval_metric = 'logloss',
    #                      objective='binary:logistic')
    
    #best_clf = RandomizedSearchCV(estimator = model,
    #                     param_distributions = params,
    #                    #scoring = 'logloss',
    #                    n_iter = 25,
    #                    verbose = 1)
    
    #best_clf.fit(X, y)
    
    #print("Best parameters:", best_clf.best_params_)
    #print("Lowest RMSE: ", (-best_clf.best_score_)**(1/2.0))
    
    

    model = XGBClassifier(n_estimators = 250,
                          base_score = 0.5,
                          eta = 0.1, 
                          max_depth = 10,  
                          learning_rate = 0.03,
                          scale_pos_weight = 1,
                          reg_lambda = 0.5,
                          min_child_weight = 1,
                          gamma = 5, #sensitive. 1 reasonable
                          reg_alpha = 0.1,
                          subsample = 0.2, #0.8 reasonable
                          colsample_bylevel = 0.8, 
                          colsample_bytree = 1, #1 reasonable
                          objective = "binary:logistic",
                          eval_metric = 'logloss',
                          #use_label_encoder = False,
                          random_state = 1,
                          #alpha = 10,
                          #silent = False, 
                          #booster = 'gbtree',
                          #tree_method = 'exact'
                          )
    
    model = model.fit(X_train, y_train)
    
    #print(clf.classes_)
    
    # plot feature importance
    #plot_importance(model)
    #plt.show()
    
    
    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    #predictions = [round(value) for value in y_pred]
    
    mse = mean_squared_error(y_test, y_pred)
    print("RMSE: %.2f" % (mse**(1/2.0)))
    
    print("Accuracy XG:", metrics.accuracy_score(y_test, y_pred).round(2))
    print("ROC AUC XG: ", roc_auc_score(y_test, y_pred).round(2))
    #print(classification_report(y_test, y_pred))
    
    # probability
    y_pred2 = model.predict_proba(X_test) 
    y_prob = np.array([x[1] for x in y_pred2]) # keep the prob for the positive class 1


    #print("Precision XG = {}".format(precision_score(y_test, y_pred, average='macro')))
    #print("Recall XG = {}".format(recall_score(y_test, y_pred, average='macro')))
    #print("Accuracy XG:", metrics.accuracy_score(y_test, y_prob))
    print("ROC AUC XG: ", roc_auc_score(y_test, y_prob))
    print(classification_report(y_test, y_pred))


    #eval_set = [(X_train, y_train), (X_test, y_test)]
    #eval_metric = ["auc","error"]
    #model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)
    
    
    # Fit model using each importance as a threshold
    # USE TO DETERMINE NUMBER OF FEATURES
    thresholds = np.sort(model.feature_importances_)
    
    #for thresh in thresholds:
        
    #    # select features using threshold
	#    selection = SelectFromModel(model, threshold=thresh, prefit=True)
	#    select_X_train = selection.transform(X_train)
	
    #    # train model
	#    selection_model = XGBClassifier()
	#    selection_model.fit(select_X_train, y_train)
	    
    #    # eval model
	#    select_X_test = selection.transform(X_test)
	#    y_pred = selection_model.predict(select_X_test)
	#    predictions = [round(value) for value in y_pred]
	#    accuracy = accuracy_score(y_test, predictions)
	#   print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
        
   
    # accuracy of classifer
    conf_matrix = pd.crosstab(
        y_test, y_pred,
        rownames = ['Actual'],
        colnames = ['Predicted'],
        margins = True,
        )

    print(conf_matrix)
    
    
    return model


def shapPlots(model):
    
    #shap.initjs()
    
    fig, ax = plt.subplots(figsize = (10,6))
    #fig.subplots_adjust(right = 0.95, bottom = 0.25, top = 0.88, left = 0.2)
    
    # shap explainer
    explainer = shap.TreeExplainer(model) 
    
    # whole dataset
    samples = X_train
    
    # individual shot
    observation = 5

    # change values to dataset or individual shot
    shap_values = explainer.shap_values(samples)    
    
    # scatter
    # interaction
   
    #shap.force_plot(explainer.expected_value, shap_values[observation,:], X_test.iloc[observation,:], 
    #               link = 'logit', matplotlib = True, show = True)
    
    # shap bar
    #shap.summary_plot(shap_values, X_test, plot_type = "bar", feature_names = X_labels, class_names = class_names)
    
    # decision plot
    #shap.decision_plot(explainer.expected_value, shap_values, X_test, link = 'logit')

    # summary plot
    shap.summary_plot(shap_values, X_train.values, feature_names = X_labels, class_names = class_names)    

    # interaction      
    #shap.dependence_plot('DIST_GOAL', shap_values, X_train, interaction_index = 'ANGLE_GOAL')
    #shap.dependence_plot('ANGLE_GOAL', shap_values, X_train, interaction_index = 'DIST_GOAL')
    #shap.dependence_plot('x_post', shap_values, X_train, interaction_index = 'y_post')
    #shap.dependence_plot('y_post', shap_values, X_train, interaction_index = 'x_post')
    #shap.dependence_plot('FOOT_VALUE', shap_values, X_train, interaction_index = 'FIELD_VALUE')
    #shap.dependence_plot('FIELD_VALUE', shap_values, X_train, interaction_index = 'FOOT_VALUE')






def price_action_plot(data, interval):
       
    fig, ax = plt.subplots(figsize = (8,12))
    ax.grid(True)
    #ax.set_ylim(10,90)
    #ax.set(yticks = np.arange(10,95,5))
    
    ax.set_xlabel('Time', labelpad = 10, fontsize = 12)
    ax.set_ylabel('RSI', labelpad = 10, fontsize = 12)
    ax.set_title('Price Action', fontsize = 12)
    

    sns.lineplot(data = data,
            #     x = data['Datetime'],
            x = data.index,
            y = interval + '_ROC_100',
            ax = ax,    
            label = interval + '_ROC_100',
            color = 'indianred',
            #alpha = 0.1,
            legend = False
            )

    
    #plt.axhline(y = 70, color = 'green', linestyle = '-')
    #plt.axhline(y = 30, color = 'red', linestyle = '-')
    
    lower, upper = 30,70
    
    
    def insertzeros(t, x, zero = 0):
        
        ta = []
        positive = (x-zero) > 0
        ti = np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
    
        for i in ti:
            y_ = np.sort(x[i:i+2])
            z_ = t[i:i+2][np.argsort(x[i:i+2])]
            t_ = np.interp(zero, y_, z_)
            ta.append( t_ )
        tnew = np.append( t, np.array(ta) )
        xnew = np.append( x, np.ones(len(ta))*zero )
        xnew = xnew[tnew.argsort()]
        tnew = np.sort(tnew)
    
        return tnew, xnew

    #t1,x1 = insertzeros(data.index, data[interval + '_RSI'], zero = lower)
    #t1,x1 = insertzeros(t1, x1, zero = upper)

    #xm = np.copy(x1)
    #xm[(x1 < lower) | (x1 > upper)] = np.nan        

    #xl = np.copy(x1)
    #xl[(x1 > lower)] = np.nan        
    #ax.plot(t1,xl, color = 'crimson')

    #xu = np.copy(x1)
    #xu[(xu < upper)] = np.nan        
    #ax.plot(t1,xu, color = 'limegreen')
    
    #ax.fill_between(data.index, data[interval + '_RSI'], 
    #                lower, where = (data[interval + '_RSI'] <= lower), 
    #                facecolor = 'crimson', 
    #                interpolate = True, 
    #                alpha = 0.5
    #                )
    
    #ax.fill_between(data.index, data[interval + '_RSI'], 
    #                upper, where = (data[interval + '_RSI'] >= upper), 
    #                facecolor = 'limegreen', 
    #                interpolate = True, 
    #                alpha = 0.5
    #               )

    
    ax2 = plt.twinx()
    ax2.set_ylabel('Price', labelpad = 10, fontsize = 12)


    sns.lineplot(data = data,
            #     x = data['Datetime'],
            x = data.index,
            y = interval + '_MA_20',
            ax = ax2,    
            label = 'MA_50',
            color = 'violet',
            alpha = 0.5,
            legend = False
            )

    sns.lineplot(data = data,
            #     x = data['Datetime'],
            x = data.index,
            y = interval + '_MA_50',
            ax = ax2,    
            label = 'MA_100',
            color = 'sienna',
            alpha = 0.5,
            legend = False
            )

    sns.lineplot(data = data,
            #     x = data['Datetime'],
            x = data.index,
            y = interval + '_MA_100',
            ax = ax2,    
            label = 'MA_200',
            color = 'indigo',
            alpha = 0.5,
            legend = False
            )

    sns.lineplot(data = data, 
                     #x = data['Datetime'],
                     x = data.index,
                     y = 'Close',
                     color = 'royalblue', 
                     ax = ax2, 
                     label = 'Price', 
                     legend = False
                     )

    fig.legend(loc = 'upper right')
    
    
    fig.subplots_adjust(right = 0.8, bottom = 0.1, top = 0.94, left = 0.1, wspace = 0.3)
    
    
    return fig


#model_XG = XG()


#shapPlots(model_XG)



price_action_plot(df, '5m')

