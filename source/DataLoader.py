from datetime import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import threading
import time
from time import sleep
import requests

from typing import List
from config import token

import collections

class Stock:
    
    def __init__(self, symbol: str, start: datetime):
        
        self.symbol = symbol

        
        try:
            self.data   = web.DataReader(symbol, 'yahoo', start).rename(str.lower, axis = "columns")
            self.error  = False
                    
        except:
            self.data   = None 
            self.error  = True
    
        
class Multithread:
    
    def __init__(self, f, iterable, threads, silently = False, independent: bool = False):
        
        if independent:
            self.output = self.get(f, iterable, threads, silently, append_func = self.list_append_independent)
        else:
            self.output = self.get(f, iterable, threads, silently, append_func = self.list_append_dependent_partly)

    
    def get_splits(self, iterable, s):
        n = len(iterable) 
        d = int(np.floor(n / s))
        
        splits =[*range(0, n-d, d)] + [n]
            
        splitted_iterable=[]
        for i in range(len(splits)):
            if i != 0:
                splitted_iterable.append(iterable[splits[i-1]:splits[i]])
                
        return splitted_iterable
    
    
    def list_append_independent(self, f, iterable, id, out_list):
        """
        Creates an empty list and then appends a 
        random number to the list 'count' number
        of times. A CPU-heavy operation!
        """
        for x in iterable:
            out_list.append(f(x))
            
    
    def list_append_dependent_partly(self,f, iterable, id, out_list):
           """
           Creates an empty list and then appends a 
           random number to the list 'count' number
           of times. A CPU-heavy operation!
           """
           out_list.append(f(iterable))

    
    def get(self, f, iterable, threads, silently, append_func):
        
        starttime = time.time()
        
        lst_sub_iterables = self.get_splits(iterable, threads)
    
        # Create a list of jobs and then iterate through
        # the number of threads appending each thread to
        # the job list 
        jobs = []
        out_list = []
        for i in range(0, threads):
            thread = threading.Thread(target=append_func(f, lst_sub_iterables[i], i, out_list))
            jobs.append(thread)
        
        # Start the threads (i.e. calculate the random number lists)
        for j in jobs:
            j.start()
    
        # Ensure all of the threads have finished
        for j in jobs:
            j.join()
    
        if silently == False:
            print('That took {} seconds'.format(time.time() - starttime))
        
        return out_list

    
class DataLoader:
                     
    def __init__(self, lst_symbols: List[str], start: datetime = datetime(2000,1,1), combine: bool = True, multiprocess: bool = True):
        
        self.symbols = lst_symbols
        self.raw_data = [Stock(sym, start) for sym in lst_symbols]
    
                
        self.excluded_sym = [x.symbol for x in self.raw_data if x.error == True]
        self.included_sym = [x for x in self.raw_data if x.error == False]
        lstStockDataClean = [x.data.close.rename(x.symbol) for x in self.raw_data if x.error == False]

        if combine & (len(lst_symbols) > 1):
            
            if len(lst_symbols) > 100:
                self.data = self.combine_stocks(lstStockDataClean, multiprocess)
            else:
                self.data = self.combine_stocks(lstStockDataClean, False)
            
        else:
            self.data = lstStockDataClean
    
    
    def combine_to_df(self, lstSeries: List[pd.Series]) -> pd.DataFrame:
        out = lstSeries[0].reset_index()
        for df in lstSeries[1:]:
            out=out.merge(df.reset_index(), on = "Date", how='outer')
            
        return out


    def combine_stocks(self, lst_Series: List[pd.Series], multiprocess: bool) -> List[pd.Series]:
              
        if multiprocess:   
            multithread= Multithread(self.combine_to_df, lst_Series, 6, silently = True, independent = False)
            out= self.combine_to_df([df.set_index("Date") for df in multithread.output])
        else:
            out=self.combine_to_df(lst_Series)
        
          
        
        return out.sort_values("Date").set_index("Date")
        
    
    def get_stock_profiles(self):
    
        out=[]
        
        for symbol in self.symbols:
            
            r = requests.get(f'https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={token}')
            
            if r.status_code == 200:
                aplProfile =r.json()
                
                if "finnhubIndustry" in aplProfile.keys():
                    industry = aplProfile["finnhubIndustry"]
                    exchange = aplProfile["exchange"]
                    currency = aplProfile["currency"]
                    
                    out.append((symbol, industry,  exchange, currency))
                else:
                    out.append((symbol, r.status_code, np.nan, np.nan))
            else:
                out.append((symbol, r.status_code, np.nan, np.nan))
                
            sleep(1)
                
        self.profiles = pd.DataFrame(out).rename(columns = dict(enumerate(["symbol", "industry", "exchange", "currency"])))
    
