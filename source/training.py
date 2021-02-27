import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date

from typing import List, Optional
#from DataLoader import DataLoader, Multithread
import pandas_datareader.data as web
from NumberOfStocksToBuy import NumberOfStocksToBuy

c25_stocks = [
    'FLS.CO',
    'ISS.CO',
    'TRYG.CO',
    'SIM.CO',
    'RBREW.CO',
    'DEMANT.CO',
    'AMBU-B.CO',
    'NETC.CO',
    'NZYM-B.CO',
    'CHR.CO',
    'NOVO-B.CO',
    'LUN.CO',
    'BAVA.CO',
    'CARL-B.CO',
    'DANSKE.CO',
    'COLO-B.CO',
    'MAERSK-B.CO',
    'MAERSK-A.CO',
    'DSV.CO',
    'VWS.CO',
    'GN.CO',
    'GMAB.CO',
    'ORSTED.CO',
    'ROCK-B.CO',
    'PNDORA.CO',
]

def pctChange(x: List[float]) -> List[float]:
    
    return pd.Series(x).pct_change().tolist()[1:]


class StockPrice:
    
    def __init__(self, date: date, price: float, volume: int):
        
        self.date = date.date() if isinstance(date, datetime) else date
        self.price = round(price, 2)
        self.volume = volume
        
        
    def __repr__(self):
        return '{date}: {price}'.format(
                date = self.date,
                price = self.price,
            )
    

class DataSplit:
    
    def __init__(self, current: StockPrice, compared: List[StockPrice]):
                
        self.current = current 
        self.compared = [p for p in compared]
       
    

class CurrPriceAndFuturePrices(DataSplit):
    
    def __init__(self, current: StockPrice, future: List[StockPrice]):
        
        DataSplit.__init__(self, current, future)
        

    def __repr__(self):
        return 'current; {curr} - future (t = {t}); {future}'.format(
                curr   = self.current,
                future = np.mean([x.price for x in self.compared]).round(2),
                t      = len(self.compared),
            )

class CurrPriceAndPastPrices(DataSplit):
    
    def __init__(self, current: StockPrice, pastPrices: List[StockPrice]):
               
        DataSplit.__init__(self, current, pastPrices)
        

    def __repr__(self):
        return 'current; {curr} - past (k = {k}); {past}'.format(
                curr   = self.current,
                past = np.mean([x.price for x in self.compared]).round(2),
                k      = len(self.compared),
            )

    

class StockProcess:
    
    def __init__(self, stockName: str, stockPrices: List[StockPrice]):
        
        self.stockName = stockName
        self.stockPrices = stockPrices
                
    
    def splitCurrPriceAndNextTDays(self, t: int) -> List[CurrPriceAndFuturePrices]:
        
        assert len(self.stockPrices) > t, 'not enough data for t = {t} (#prices = {n})'.format(t = t, n = len(self.stockPrices))
        
        dataSplit = list()
        
        for i in range(len(self.stockPrices)):
           
            current = self.stockPrices[i]
            future  = self.stockPrices[i+1: i+1+t]
            
            if len(future) == t:
            
                dataSplit.append(CurrPriceAndFuturePrices(current, future))
            
        return dataSplit
            

    def splitCurrPriceAndPastKDays(self, k: int) -> List[CurrPriceAndPastPrices]:
        
        assert len(self.stockPrices) > k, 'not enough data for k = {k} (#prices = {n})'.format(k = k, n = len(self.stockPrices))
        
        dataSplit = list()
        
        for i in range(len(self.stockPrices)):
           
            current = self.stockPrices[i]
            past    = self.stockPrices[i - k: i]
            
            if len(past) == k:
            
                dataSplit.append(CurrPriceAndPastPrices(current, past))
            
        return dataSplit
            

class Featues:
    
    def __init__(self, date: date, values: List[float]):
        self.date   = date
        self.values = values
        
class Measurement:
    
    def __init__(self, date: date, value: float):
        self.date   = date
        self.value = value
        
        
        
class TrainingData:
    
    def __init__(self, stockProcess: StockProcess, k: int, t: int, p: float):
        
        self.stockName       = stockProcess.stockName
        self.stockPrices     = stockProcess.stockPrices
        self.currAndFuture   = stockProcess.splitCurrPriceAndNextTDays(t = t)
        self.currAndPast     = stockProcess.splitCurrPriceAndPastKDays(k = k)
        
        self.nFutureDaysMoreExpensiveThanCurrentPrice = self.countDaysMoreExpensiveThanCurrent(self.currAndFuture, p = p)
        self.nPastDaysMoreExpensiveThanCurrentPrice   = self.countDaysMoreExpensiveThanCurrent(self.currAndPast, p = p)
        
        self.futurePrices = self.comparedPricesPerCurrentDate(self.currAndFuture)
        self.pastPrices   = self.comparedPricesPerCurrentDate(self.currAndPast)
    
    def countDaysMoreExpensiveThanCurrent(self, dataPairList: List[DataSplit], p: float) -> List[Measurement]:
                
        measurements = list()
        
        for dataPair in dataPairList:
            
            measurement = sum(1 for x in dataPair.compared if dataPair.current.price * (1 + p) < x.price)
            measurements.append(Measurement(dataPair.current.date, measurement))
            
        return measurements
            
        
        
    
    def pctPriceChange(self, dataPairList: List[CurrPriceAndPastPrices]):
        
        features = list()
        
        for dataPair in dataPairList:
            
            featureValues = pctChange([x.price for x in datasplit.compared] + [datasplit.current.price])    
            features.append(Feature(dataPair.current.date, featureValues))
            
        return features
            
            
        
        #self.featuresFromPast= [[datasplit.current.date] + pctChange([x.price for x in datasplit.compared] + [datasplit.current.price]) for datasplit in self.currAndPast]
        #profitDaysPastCount= [(datasplit.current.date, datasplit.countDaysMoreExpensiveThanCurrentPricePlusPct(p = p)) for datasplit in self.currAndPast]
        
        
        
    def self.nPastDaysMoreExpensiveThanCurrentPrice
        self.featuresFromPast = list()
        i = 0
        
        for i in range(len(profitDaysPastCount)):
                       
            if i >= k:
                
                currentDate, currentCount = profitDaysPastCount[i]
                pastCounts                = [x[1] for x in profitDaysPastCount[i - k: i]]
                
                self.featuresFromPast.append([currentDate] + pastCounts)
                    
            
            i =+ 1
        
        
    def comparedPricesPerCurrentDate(self, dataPairList: List[DataSplit]):
    
        comparedPrices = dict()
        
        for dataPair in dataPairList:
            
            currentDate = dataPair.current.date
            comparedPrices[currentDate] = [x.price for x in dataPair.compared]

        return comparedPrices


stockName = 'NOVO-B.CO'


data = web.get_data_yahoo(stockName, start = datetime(2011,1,1))\
            .reset_index()\
            .rename(str.lower, axis = "columns")\
            .loc[:,['date', 'close', 'volume']]

stockPrices = [StockPrice(*s) for s in data.values]
sp = StockProcess(stockName, stockPrices)
td = TrainingData(sp, 30, 30, 0.02)

td.featuresFromPast


# Split data



# Training
labelsDf   = pd.DataFrame(td.profitDaysCount).rename(columns = {0: 'date', 1: 'label'})
featuresDf = pd.DataFrame(td.featuresFromPast).rename(columns = {0: 'date'})

finalDf = featuresDf.merge(labelsDf, on = 'date', how = 'left').set_index('date').query("label.notna()")

X = finalDf.drop(columns = ['label']).copy()
y = finalDf.label.copy()


# X_train = X.copy().iloc[:-30]
# y_train = y.copy().iloc[:-30]
# X_test  = X.copy().iloc[-30:]
# y_test  = y.copy().iloc[-30:]

from sklearn.model_selection import train_test_split# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)



# Import the model we are using
from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
rf.fit(X_train, y_train);

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)



summaryPred = pd.DataFrame(zip(y_pred_lr, y_pred_rf, y_test), index = y_test.index).round().merge(pd.DataFrame([p.__dict__ for p in td.stockPrices]).set_index('date'), left_index = True, right_index = True)
summaryPred['future'] = summaryPred.index.map(td.pricesFromFuture.get)
summaryPred['past']  = list(X_test.values)




pricesForUnlabeled = pd.DataFrame([{'date': stock.date, 'price': stock.price} for stock in td.stockPrices[-30:]]).set_index('date')

featuresForUnlabeled = pd.DataFrame(td.featuresFromPast[-30:]).rename(columns = {0: 'date'}).set_index('date')
pd.DataFrame(zip(featuresForUnlabeled.index, lr.predict(featuresForUnlabeled), pricesForUnlabeled.price))\
            .rename(columns = dict(enumerate(['date', 'pred', 'current'])))\
            .assign(future = lambda df: df.date.map(td.pricesFromFuture.get))
            
pd.DataFrame(zip(featuresForUnlabeled.index, rf.predict(featuresForUnlabeled), pricesForUnlabeled.price))\
            .rename(columns = dict(enumerate(['date', 'pred', 'current'])))\
            .assign(future = lambda df: df.date.map(td.pricesFromFuture.get))



td.stockPrices


sum((y_pred_lr.round() - y_test) ** 2)
sum((y_pred_rf.round() - y_test) ** 2)

from collections import Counter
Counter([s.countDaysWithPotentialProfit() for s in sp])

for x in sp:

    print(sum(1 for p in x.futurePrices if x.currentPrice.priceClose < p.priceClose))



        
                



stockDf = list()

for stockName in c25_stocks:
    
    print(stockName)
    
    try:
        data = web.get_data_yahoo(stockName, start = datetime(2000,1,1))\
                    .reset_index()\
                    .rename(str.lower, axis = "columns")\
                    .loc[:,['date', 'low', 'high', 'open', 'close', 'volume']]
        
        stockPrices = [StockPrice(*s) for s in data.values]
        
        
        pctChange = data.set_index('date').pct_change().reset_index()
        stockPricesPct = [StockPrice(*s) for s in data.values]
        
        
        
        
    except:
        
        stockPricesDay = []
        

    stockDf.append()



danskeBank = web.get_data_yahoo('DANSKE.CO', start = datetime(2000,1,1)).rename(str.lower, axis = "columns")

data = web.get_data_yahoo('DANSKE.CO', start = datetime(2011,1,1))\
            .reset_index()\
            .rename(str.lower, axis = "columns")\
            .loc[:,['date', 'low', 'high', 'open', 'close', 'volume']]

stockPrices = [StockPrice(*s) for s in data.values]


pctChange = data.set_index('date').pct_change().reset_index()
stockPricesPct = [StockPrice(*s) for s in data.values]
pricesFromFuture


# 1800: 367358