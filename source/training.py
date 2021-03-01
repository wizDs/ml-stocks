import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date

from typing import List, Optional, Mapping
#from DataLoader import DataLoader, Multithread
import pandas_datareader.data as web
from NumberOfStocksToBuy import NumberOfStocksToBuy

from Class.TrainingDataGenerator import TrainingDataGenerator
from Class.TrainingData import TrainingData
from Class.ModelPrediction import ModelPrediction, ModelScore

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



tdGenerator = TrainingDataGenerator()
td = tdGenerator.byStockName('GN.CO', start = date(2011, 1, 1))

evaluationDataList = [td.splitDataAtIndex(index = i) for i in range(1000, len(td.X) + 1)]

ed=evaluationDataList[1000]

rf = ModelPrediction(ed)    
pred = rf.predictSeries(ed.X_test)

import json
ModelScore(ed.currentDate, pred).toJson(f'{ed.currentDate}.txt')






listPredictions = list()

for ed in evaluationDataList:
    
    rf = ModelPrediction(ed)    
    rf.toPickle(f'Models/{ed.currentDate}.pickle')

    pred = rf.predictSeries(ed.X_test)
    listPredictions.append(pred)
    


z=pd.DataFrame(listPredictions)
z['Price']= z.index.map(td.stockPriceMapper)
z['RegressionForrest'] = z.eval('sqrt(RegressionForrest / 30) * 10')

#z.to_csv('')

a = z.copy()
a = a[pd.Series(a.index).between(date(2016, 1, 1), date(2016, 5, 1)).values]


import matplotlib.pyplot as plt
import seaborn as sns

# create figure and axis objects with subplots()
fig, ax = plt.subplots(dpi = 150)
# make a plot
sns.lineplot(a.index, a.Price, color="red")
# set x-axis label
ax.set_xlabel("",fontsize=14)
# set y-axis label
ax.set_ylabel("Stock Price (GN Store)",color="red",fontsize=14)

ax.set_title("High model score means high recommendation for buying")
plt.xticks(rotation=90)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
sns.lineplot(a.index, a["RegressionForrest"],color="blue")
ax2.set_ylabel("Model score",color="blue",fontsize=14)

plt.show()

# save the plot as a file
fig.savefig('model_performance.jpg',
            format='jpeg',
            dpi=200,
            bbox_inches='tight')




# Training
X = td.X.copy()
y = td.y.copy()

# Split data
from sklearn.model_selection import train_test_split# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


# random forrest
from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
rf.fit(X_train, y_train);
y_pred_rf = rf.predict(X_test)



summaryPred = pd.DataFrame(zip(y_pred_lr, y_pred_rf, y_test), index = y_test.index).round().merge(pd.DataFrame([p.__dict__ for p in td.stockPrices]).set_index('date'), left_index = True, right_index = True)
summaryPred['future'] = summaryPred.index.map(td.futurePrices.get)
summaryPred['past']  = list(X_test.values)

summaryPred.to_csv('temp.csv', sep = ';')

# Unlabeled
from functools import partial
def mapToScore(moreExpensiceThanCurrentPriceCount: int, t: int) -> int:
    return moreExpensiceThanCurrentPriceCount / t * 10


X_unlabeled = td.unlabeledData.copy().drop(columns = ['label'])

# linear regression
y_pred_lr_unlabeled = lr.predict(X_unlabeled)

# random forrest
y_pred_rf_unlabeled = rf.predict(X_unlabeled)

summaryUnlabeled = pd.DataFrame({
     'Date': X_unlabeled.index,
     #'LogisticRegression': map(partial(mapToScore, t = 30), y_pred_lr_unlabeled),
     'LogisticRegression': y_pred_lr_unlabeled.round(),
     'RandomForrest': y_pred_rf_unlabeled.round(),
     'Price' : X_unlabeled.index.map(td.stockPriceMapper)
 })

summaryUnlabeled



sum((y_pred_lr.round() - y_test) ** 2)
sum((y_pred_rf.round() - y_test) ** 2)


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