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
            

class Features:
    
    def __init__(self, date: date, values: List[float]):
        self.date   = date
        self.values = values
        
    def prepareForPd(self):
        
        featuresWithCurrentDate = {'date': self.date}
        for i, featureValue in enumerate(self.values):
            featuresWithCurrentDate[i] = featureValue
            
        return featuresWithCurrentDate
        
class Measurement:
    
    def __init__(self, date: date, value: float):
        self.date   = date
        self.value  = value
        
    def prepareForPd(self):
        
        return {'date': self.date, 'label': self.value}
        
class TrainingData:
    
    def __init__(self, stockProcess: StockProcess, k: int, t: int, p: float):
        
        self.stockName       = stockProcess.stockName
        self.stockPrices     = stockProcess.stockPrices
        
        # Future
        currAndFuture   = stockProcess.splitCurrPriceAndNextTDays(t = t)        
        self.nFutureDaysMoreExpensiveThanCurrentPrice = self.countDaysMoreExpensiveThanCurrent(currAndFuture, p = p)
        self.futurePrices    = self.comparedPricesPerCurrentDate(currAndFuture)
        
        # Past
        currAndPast     = stockProcess.splitCurrPriceAndPastKDays(k = k)
        self.nPastDaysMoreExpensiveThanCurrentPrice   = self.countDaysMoreExpensiveThanCurrent(currAndPast, p = p)
        self.pastPrices      = self.comparedPricesPerCurrentDate(currAndPast)
        self.featuresForCurrentPrice = self.splitCurrentEntityAndPast(self.nPastDaysMoreExpensiveThanCurrentPrice, k = k)
    
        # For each date, join the past (features) and the future (label)
        dataframe          = self.joinFeaturesAndLabels()
        self.labeledData   = dataframe.query("label.notna()")
        self.unlabeledData = dataframe.query("label.isna()")
        
        # split into features (X) and labels (y)
        self.X = self.labeledData.drop(columns = ['label']).copy()
        self.y = self.labeledData.label.copy()
    
    def countDaysMoreExpensiveThanCurrent(self, dataPairList: List[DataSplit], p: float) -> List[Measurement]:
                
        measurements = list()
        
        for dataPair in dataPairList:
            
            measurement = sum(1 for x in dataPair.compared if dataPair.current.price * (1 + p) < x.price)
            measurements.append(Measurement(dataPair.current.date, measurement))
            
        return measurements
            
            
        
    def splitCurrentEntityAndPast(self, measurements: List[Measurement], k: int) -> List[Features]:
        
        i = 0
        pastMeasurements = list()
        
        for i in range(len(measurements)):
                       
            if i >= k:
                
                currentDate  = measurements[i].date
                pastCounts   = [x.value for x in measurements[i - k + 1: i + 1]]
                
                pastMeasurements.append(Features(currentDate, pastCounts))
                    
            
            i =+ 1
            
        return pastMeasurements
        
        
    def comparedPricesPerCurrentDate(self, dataPairList: List[DataSplit]):
    
        comparedPrices = dict()
        
        for dataPair in dataPairList:
            
            currentDate = dataPair.current.date
            comparedPrices[currentDate] = [x.price for x in dataPair.compared]

        return comparedPrices
    
    def joinFeaturesAndLabels(self):
        
        labelsDf   = pd.DataFrame([m.prepareForPd() for m in self.nFutureDaysMoreExpensiveThanCurrentPrice])
        featuresDf = pd.DataFrame([f.prepareForPd() for f in self.featuresForCurrentPrice])
        
        finalDf = featuresDf.merge(labelsDf, on = 'date', how = 'left').set_index('date')
        
        return finalDf
    
    def pctPriceChange(self, dataPairList: List[CurrPriceAndPastPrices]):
        
        features = list()
        
        for dataPair in dataPairList:
            
            featureValues = pctChange([x.price for x in dataPair.compared] + [dataPair.current.price])    
            features.append(Features(dataPair.current.date, featureValues))
            
        return features
            


stockName = 'NOVO-B.CO'


data = web.get_data_yahoo(stockName, start = datetime(2011,1,1))\
            .reset_index()\
            .rename(str.lower, axis = "columns")\
            .loc[:,['date', 'close', 'volume']]

stockPrices = [StockPrice(*s) for s in data.values]
sp = StockProcess(stockName, stockPrices)
td = TrainingData(sp, 30, 30, 0.02)


# Training
X = td.X.copy()
# X['season'] = X.index.map(lambda x: x.month).astype(str)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop = 'first')
months = X.index.map(lambda x: x.month).values.reshape(-1 ,1)
ohe.fit(months)
season = pd.DataFrame(ohe.transform(months).toarray(), index = X.index)
y = td.y.copy()

# Split data
from sklearn.model_selection import train_test_split# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# linear regression
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
summaryPred['future'] = summaryPred.index.map(td.futurePrices.get)
summaryPred['past']  = list(X_test.values)

summaryPred.to_csv('temp.csv', sep = ';')



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