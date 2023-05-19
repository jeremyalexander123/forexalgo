import numpy as np
import pandas as pd
import talib
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pytz
import schedule
import time
from threading import Thread
from time import sleep
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

from scipy.signal import argrelextrema
from collections import deque
from functools import reduce


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(precision = 5)

# current date
#date = datetime.today().strftime('%Y-%m-%d')
my_date = datetime.datetime.now(pytz.timezone('Etc/GMT-12'))
prev_12hrs = datetime.datetime.now(pytz.timezone('Etc/GMT+12'))
prev_24hrs = my_date - datetime.timedelta(hours = 24, minutes = 0)

print(my_date)
print(prev_24hrs)

#my_date = datetime.datetime.now() 
#prev_24hrs = my_date - datetime.timedelta(hours = 24, minutes = 0)


#print(my_date)
#print(prev_24hrs)

#print(datetime.datetime.now())

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
    
    data = data.rename(columns = {'Adj Close' : 'Adj_Close'})
    
    # moving average
    #data[interval + '_MA_20'] = talib.SMA(data['Adj Close'].values, timeperiod = 20)
    #data[interval + '_MA_50'] = talib.SMA(data['Adj Close'].values, timeperiod = 50)
    #data[interval + '_MA_100'] = talib.SMA(data['Adj Close'].values, timeperiod = 100)
 
    # rate of change
    #data[interval + '_ROC_20'] = talib.ROC(data['Adj Close'].values, timeperiod = 20).round(1)
    #data[interval + '_ROC_50'] = talib.ROC(data['Adj Close'].values, timeperiod = 50).round(1)
    #data[interval + '_ROC_100'] = talib.ROC(data['Adj Close'].values, timeperiod = 100).round(1)

    # rsi
    data[interval + '_RSI'] = talib.RSI(data['Adj_Close'].values, timeperiod = 14)
  
    # overbought/oversold for given timeframe
    data[interval + '_OB_OS'] = np.where(data.iloc[:,-1] <= 30, "OS", 
                                np.where(data.iloc[:,-1] >= 70, "OB", np.NaN      
                                ))
    
    # drop data to be replace with 1min data
    #data = data.drop(['Open','High','Low','Close','Adj Close','Volume'], axis = 1)

    # merge 1 min and given time interval
    #df = One_min.merge(data, how = 'left', on = ['Datetime'])
    

    return data


#df_out = data_metrics('EURUSD=X', '2022-10-06', '2022-10-07', '5m')
df_out = data_metrics('EURUSD=X', prev_24hrs, my_date, '5m')

#print(df_out[0:60])

def getHigherHighs(data, order = 1, K = 2):
    
  '''
  Finds consecutive higher highs in price pattern.
  Must not be exceeded within the number of periods indicated by the width 
  parameter for the value to be confirmed.
  K determines how many consecutive highs need to be higher.
  '''
  
  # Get highs
  # indexes
  high_idx = argrelextrema(data, np.greater, order=order)[0]
  
  # prices
  highs = data[high_idx]
  
  # Ensure consecutive highs are higher than previous highs
  extrema = []
  
  ex_deque = deque(maxlen = K)
  
  for i, idx in enumerate(high_idx):
      
    if i == 0:
        
      ex_deque.append(idx)
      
      continue
  
    if highs[i] < highs[i-1]:
        
      ex_deque.clear()
      
    ex_deque.append(idx)
    
    if len(ex_deque) == K:
        
      extrema.append(ex_deque.copy())
  
  return extrema



def getLowerHighs(data, order = 1, K = 2):
    
  '''
  Finds consecutive higher highs in price pattern.
  Must not be exceeded within the number of periods indicated by the width 
  parameter for the value to be confirmed.
  K determines how many consecutive highs need to be higher.
  '''
  
  # Get highs
  # indexes
  high_idx = argrelextrema(data, np.greater, order=order)[0]
  
  # prices
  highs = data[high_idx]
  
  # Ensure consecutive highs are higher than previous highs
  extrema = []
  
  ex_deque = deque(maxlen = K)
  
  for i, idx in enumerate(high_idx):
      
    if i == 0:
        
      ex_deque.append(idx)
      
      continue
  
    if highs[i] > highs[i-1]:
        
      ex_deque.clear()
      
    ex_deque.append(idx)
    
    if len(ex_deque) == K:
        
      extrema.append(ex_deque.copy())
  
  return extrema







# Get higher highs, lower lows, etc.
order = 12
hh_pr = getHigherHighs(df_out['Close'].values, order)
lh_pr = getLowerHighs(df_out['Close'].values, order)
#ll_pr = getLowerLows(price, order)
#hl_pr = getHigherLows(price, order)


hh_rsi = getHigherHighs(df_out['5m_RSI'].values, order)
lh_rsi = getLowerHighs(df_out['5m_RSI'].values, order)
#ll = getLowerLows(price, order)
#hl = getHigherLows(price, order)

#print(hh)
#print(df_out[0:60])

#print(df_out[60:120])

#print(df_out[120:180])


hh_pr_df = pd.DataFrame([i[1] for i in hh_pr])
hh_pr_df['Price_Action'] = 'HH_Price'
hh_pr_df = hh_pr_df.set_index(0)

lh_pr_df = pd.DataFrame([i[1] for i in lh_pr])
lh_pr_df['Price_Action'] = 'LH_Price'
lh_pr_df = lh_pr_df.set_index(0)


pr_df = pd.concat([hh_pr_df, lh_pr_df]).sort_index()
pr_df = pr_df[~pr_df.index.duplicated(keep='first')]


hh_rsi_df = pd.DataFrame([i[1] for i in hh_rsi])
hh_rsi_df['RSI_Action'] = 'HH_RSI'
hh_rsi_df = hh_rsi_df.set_index(0)

lh_rsi_df = pd.DataFrame([i[1] for i in lh_rsi])
lh_rsi_df['RSI_Action'] = 'LH_RSI'
lh_rsi_df = lh_rsi_df.set_index(0)


rsi_df = pd.concat([hh_rsi_df, lh_rsi_df]).sort_index()
rsi_df = rsi_df[~rsi_df.index.duplicated(keep='first')]

#print(pr_df)

#print(lh_pr_df)


#print(rsi_df)


#print(df_out[250:])

pdList = [df_out, pr_df, rsi_df] 
new_df = pd.concat(pdList, axis = 1)



print(new_df[0:60])
print(new_df[60:120])





def plot():
    
    fig, ax = plt.subplots(figsize = (8,12))
    ax.grid(True)
    #ax.set_ylim(10,90)
    #ax.set(yticks = np.arange(10,95,5))
    
    ax.set_xlabel('Time', labelpad = 10, fontsize = 12)
    ax.set_ylabel('RSI', labelpad = 10, fontsize = 12)
    ax.set_title('Price Action', fontsize = 12)
    
    #price = new_df.dropna(subset = 'Price_Action')
    price = new_df[new_df['Price_Action'] == 'HH_Price']
    
    #rsi = new_df.dropna(subset = 'RSI_Action')
    rsi = new_df[new_df['RSI_Action'] == 'LH_RSI']
    #dates = df_out.index
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    print(price)
    print(rsi)


    sns.lineplot(data = df_out,
            #     x = data['Datetime'],
            x = df_out.index,
            y = df_out['Close'],
            #y = interval + '_ROC_100',
            ax = ax,    
            #label = interval + '_ROC_100',
            color = 'indianred',
            alpha = 0.1,
            legend = False
            )
    
    plt.scatter(price.index, price['Close'], marker = '^', c = 'purple')

    #plt.scatter(dates[lh_idx], price[lh_idx-order], marker='v', c=colors[2])
    #plt.scatter(dates[ll_idx], price[ll_idx-order], marker='v', c=colors[3])
    #plt.scatter(dates[hl_idx], price[hl_idx-order], marker='^', c=colors[4])
    #_ = [plt.plot(dates[i], price[i], c=colors[1]) for i in hh]
    #_ = [plt.plot(dates[i], price[i], c=colors[2]) for i in lh]
    #_ = [plt.plot(dates[i], price[i], c=colors[3]) for i in ll]
    #_ = [plt.plot(dates[i], price[i], c=colors[4]) for i in hl]
    
    
    ax2 = plt.twinx()
    ax2.set_ylabel('RSI', labelpad = 10, fontsize = 12)
    
    plt.scatter(rsi.index, rsi['5m_RSI'], marker = 'x', c = 'green')
    
    sns.lineplot(data = df_out,
            #     x = data['Datetime'],
            x = df_out.index,
            y = df_out['5m_RSI'],
            #y = interval + '_ROC_100',
            ax = ax2,    
            #label = interval + '_ROC_100',
            color = 'blue',
            alpha = 0.1,
            legend = False
            )
    

    plt.xlabel('Date')
    
    #plt.scatter(dates[hh_idx], price[hh_idx-order], marker = '^', c = 'blue')

plot()



import pandas as pd
import requests


response = requests.get('https://api.covid19api.com/summary')
covid_data = pd.json_normalize(response.json())

google_password = 'ategvrxmlaurbqgj'


def send_tradeNotification(send_to, subject, df):
    
    # google account and password
    send_from = 'jeremyalexander60@gmail.com'
    password = google_password
    
    # email message 
    message = """\
    <p><strong>Trade Setup&nbsp;</strong></p>
    <p>
    
    <br>
    
    </p>
    <p><strong>-&nbsp;
    
    </strong><br><strong>JA&nbsp;    </strong></p>
    
    """
    
    for receiver in send_to:
        multipart = MIMEMultipart()
        multipart['From'] = send_from
        multipart['To'] = receiver
        multipart['Subject'] = subject  
        attachment = MIMEApplication(df.to_csv())
        attachment['Content-Disposition'] = 'attachment; filename=" {}"'.format(f'{subject}.csv')
        multipart.attach(attachment)
        multipart.attach(MIMEText(message, 'html'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(multipart['From'], password)
        server.sendmail(multipart['From'], multipart['To'], multipart.as_string())
        server.quit()


#send_tradeNotification(['jeremyalexander60@gmail.com'], 'Trade Setup', df)




