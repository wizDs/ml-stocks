import pandas_datareader.data as web
import pandas as pd
from datetime import date
from typing import Optional
from Class.PriceType import PriceType
from Class.StockPrice import StockPrice
from Class.StockProcess import StockProcess
from Class.TrainingData import TrainingData

class TrainingDataGenerator:
    
    k = 30
    t = 30
    p = 0.02


    def byStockName(cls, name: str, start: Optional[date] = None, priceType: PriceType = PriceType.Open) -> TrainingData:
        
        
        data = web.get_data_yahoo(name, start = start)\
                    .reset_index()\
                    .rename(str.lower, axis = "columns")\
                    .loc[:,['date', priceType.value, 'volume']]
        
        stockPrices = [StockPrice(*s) for s in data.values]
        stockProcess = StockProcess(name, stockPrices)
        trainingData = TrainingData(stockProcess, cls.k, cls.t, cls.p)
        
        return trainingData


    def byCsvFile(cls, path: str, **kwargs) -> TrainingData:
        
        data = pd.read_csv(path, **kwargs)
        name = path.split("/")[-1].replace(".csv", "")
        
        stockPrices = [StockPrice(*s) for s in data.values]
        stockProcess = StockProcess(name, stockPrices)
        trainingData = TrainingData(stockProcess, cls.k, cls.t, cls.p)
        
        return trainingData
        
        
        