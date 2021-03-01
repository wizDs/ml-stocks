import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from typing import List, Optional
from sklearn.preprocessing import OneHotEncoder

from .StockProcess import StockProcess
from .DataPair import DataPair
from .Measurement import Measurement
from .Features import Features

class EvaluationData:
    
    def __init__(self, pastAndCurrentData: pd.DataFrame, currentData: pd.DataFrame, t: int):
        
        
        self.X_train     = pastAndCurrentData.drop(columns =['label'])
        self.y_train     = pastAndCurrentData.label
        self.X_test      = currentData.drop(index = ['label'])
        self.y_test      = currentData.label
        
        self.t           = t
        self.currentDate = self.X_test.name
        
    def __repr__(self):
        
        emptySet = len(self.X_train) > 0
        start    = min(self.X_train.index) if emptySet else ''
        end      = max(self.X_train.index) if emptySet else ''
        
        
        return 'EvaluationData(current date: {currentDate}, training period: {period})'.format(
                    currentDate   = self.currentDate,
                    period         = f'{start} - {end}' if emptySet else 'None',
                )


class TrainingData:
    
    def __init__(self, stockProcess: StockProcess, k: int, t: int, p: float):
        
        self.t = t
        self.stockName       = stockProcess.stockName
        self.stockPrices     = stockProcess.stockPrices
        self.stockPriceMapper= stockProcess.getMapper()
        
        # Future (labels)
        currAndFuture   = stockProcess.splitCurrPriceAndNextTDays(t = t)        
        self.nFutureDaysMoreExpensiveThanCurrentPrice = self.countDaysMoreExpensiveThanCurrent(currAndFuture, p = p)
        self.futurePrices    = self.comparedPricesPerCurrentDate(currAndFuture)
        
        # Past (features)
        currAndPast     = stockProcess.splitCurrPriceAndPastKDays(k = k)
        self.nPastDaysMoreExpensiveThanCurrentPrice   = self.countDaysMoreExpensiveThanCurrent(currAndPast, p = p)
        self.pastPrices      = self.comparedPricesPerCurrentDate(currAndPast)
        self.featuresForCurrentPrice = self.splitCurrentEntityAndPast(self.nPastDaysMoreExpensiveThanCurrentPrice, k = k)
    
        # For each date, join the past (features) and the future (label)
        dataframe          = self.joinFeaturesAndLabels()
        
        # Merge seasonality component into training data frame
        seasonality        = self.seasonalityDummies(dataframe.index).rename(mapper=lambda m: 'm{}'.format(m), axis = 1)
        dataframe = dataframe.merge(seasonality, left_index = True, right_index = True, how = 'left')
        
        # Add trend feature
        dataframe['trend'] = self.trendFeature(dataframe.index)
        
        # split labeled and unlabeled data
        self.labeledData   = dataframe.query("label.notna()")
        self.unlabeledData = dataframe.query("label.isna()")
        
        # split into features (X) and labels (y)
        self.X = self.labeledData.drop(columns = ['label']).copy()
        self.y = self.labeledData.label.copy()
    
    def countDaysMoreExpensiveThanCurrent(self, dataPairList: List[DataPair], p: float) -> List[Measurement]:
                
        measurements = list()
        
        for dataPair in dataPairList:
            
            measurement = sum(1 for x in dataPair.compared if dataPair.current.price * (1 + p) < x.price)
            measurements.append(Measurement(dataPair.current.date, measurement))
            
        return measurements
            
            
        
    def splitCurrentEntityAndPast(self, measurements: List[Measurement], k: int) -> List[Features]:
        
        listOfFeatures = list()
        
        for i in range(len(measurements)):
                       
            if i >= k:
                
                currentDate      = measurements[i].date
                pastMeasurements = measurements[i - k + 1: i + 1]
                
                listOfFeatures.append(Features(currentDate, pastMeasurements))
                    
            
        return listOfFeatures
        
        
    def comparedPricesPerCurrentDate(self, dataPairList: List[DataPair]):
    
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
    
    def pctPriceChange(self, dataPairList: List[DataPair]):
        
        features = list()
        
        for dataPair in dataPairList:
            
            pastAndCurrentPrices = [x.price for x in dataPair.compared] + [dataPair.current.price]
            measurementValues = pd.Series(pastAndCurrentPrices).pct_change()
            measurements      = [Measurement(d, m) for d, m in zip([x.date for x in dataPair.compared], measurementValues)]
                        
            features.append(Features(dataPair.current.date, measurements))
            
        return features
            
    def seasonalityDummies(self, dates: List[date]) -> pd.DataFrame:
        
        months = pd.DataFrame({'month': [d.month for d in dates]})
        ohe    = OneHotEncoder(drop = 'first')
        season = ohe.fit_transform(months).toarray()
        
        return pd.DataFrame(season, index = dates, columns = np.sort(months.month.unique())[1:]).astype(int)
    
    def trendFeature(self, dates: List[date]) -> List[int]:
        return [d.year - 2000 for d in dates]


    def plotTimeSeries(self, title: Optional[str] = None):
        
        dates  = self.labeledData.index.copy()
        prices = dates.map(self.stockPriceMapper)
        
        # rename the column with symbole name
        price = pd.DataFrame({self.stockName: prices}, index = dates)
        ax = price.plot(title = title)
        ax.set_xlabel('date')
        ax.set_ylabel('closing price')
        ax.grid()
        plt.show()
    
    def splitDataAtIndex(self, index: int) -> EvaluationData:
        
        data = self.labeledData.copy()
        
        availableDataAtTimeX         = data.iloc[:index+1]        
        availableTrainingDataAtTimeX = availableDataAtTimeX.iloc[:-(self.t)]
        currentData                  = availableDataAtTimeX.iloc[-1]
        
        return EvaluationData(availableTrainingDataAtTimeX, currentData, self.t)
        
        