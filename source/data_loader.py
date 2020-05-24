# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:21:15 2020

@author: Wilson
"""

import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import threading
import time
from time import sleep
import requests

from typing import List
from config import config

import collections
stock_data_class = collections.namedtuple("stock_data", ["data", "symbol", "error"])
stock_data_class_all = collections.namedtuple("stock_data", ["data", "lst_series_data", "excluded_symbols"])

# =============================================================================
# 
# =============================================================================
class multithread:
    
    def get_splits(iterable, s):
        n = len(iterable) 
        d = int(np.floor(n / s))
        
        splits =[*range(0, n-d, d)] + [n]
            
        splitted_iterable=[]
        for i in range(len(splits)):
            if i != 0:
                splitted_iterable.append(iterable[splits[i-1]:splits[i]])
                
        return splitted_iterable
    
    
    def list_append_independent(f, iterable, id, out_list):
        """
        Creates an empty list and then appends a 
        random number to the list 'count' number
        of times. A CPU-heavy operation!
        """
        for x in iterable:
            out_list.append(f(x))
            
    
    def list_append_dependent_partly(f, iterable, id, out_list):
           """
           Creates an empty list and then appends a 
           random number to the list 'count' number
           of times. A CPU-heavy operation!
           """
           out_list.append(f(iterable))

    
    def get(f, iterable, threads, silently, append_func):
        
        starttime = time.time()
        
        lst_sub_iterables = multithread.get_splits(iterable, threads)
    
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

    def get_independently(f, iterable, threads, silently = False):
        return multithread.get(f, iterable, threads, silently, append_func = multithread.list_append_independent)
    
    def get_dependently(f, iterable, threads, silently = False): 
        return multithread.get(f, iterable, threads, silently, append_func = multithread.list_append_dependent_partly)


# =============================================================================
# 
# =============================================================================

class data_loader:
    
    def get_stock_data(symbol: str, start = dt.datetime(2000,1,1)):
        
        try:
            return stock_data_class(web.DataReader(symbol,'yahoo',start)\
                                        .rename(str.lower, axis = "columns"), symbol, False)
        except:
            return stock_data_class(None, symbol, True)
        
    def singlethread_map(f, iterable):
        starttime = time.time()
        output = [*map(f, iterable)]
        print('That took {} seconds'.format(time.time() - starttime))
        
        return output
    
        
    def combine_stock_data(lstSeries):
        out = lstSeries[0].reset_index()
        for df in lstSeries[1:]:
            out=out.merge(df.reset_index(), on = "Date", how='outer')
            
        return out


    def combine_stock_data_from_lst(lst_Series: List[pd.Series], multiprocess):
        
        starttime = time.time()
        
        if multiprocess:   
            out= multithread.get_dependently(data_loader.combine_stock_data, lst_Series, 6, silently = True)
            out= data_loader.combine_stock_data([df.set_index("Date") for df in out])
        else:
            out=data_loader.combine_stock_data(lst_Series)
        
        print('That took {} seconds'.format(time.time() - starttime))
            
        
        return out.sort_values("Date").set_index("Date")
        
        
    def get_stock_data_from_lst(lst_symbols: List[str], start = dt.datetime(2000,1,1), combine = True, multiprocess = True):
        
        if multiprocess:
            lstStockData = multithread.get_independently(data_loader.get_stock_data, lst_symbols, 6)
        else:
            lstStockData = data_loader.singlethread_map(data_loader.get_stock_data, lst_symbols)
                
        lstStockDataError = [x.symbol for x in lstStockData if x.error == True]
        lstStockDataFilter= [x for x in lstStockData if x.error == False]
        lstStockDataClean = [x.data.close.rename(x.symbol) for x in lstStockData if x.error == False]

        if combine:
            combined_data=data_loader.combine_stock_data_from_lst(lstStockDataClean, multiprocess)
            return stock_data_class_all(combined_data, lstStockDataFilter, lstStockDataError)
        else:
            return stock_data_class_all(lstStockDataClean, lstStockDataFilter, lstStockDataError)
    
    def get_stock_profile(lst_stock_data_class: List[stock_data_class]):
    
        out=[]
        
        for df, symbol, error in lst_stock_data_class:
            
            r = requests.get(f'https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={config.token}')
            
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
                
        return pd.DataFrame(out).rename(columns = dict(enumerate(["symbol", "industry", "exchange", "currency"])))
    
    def multithread_vs_singlethread(data, n):
        print(f"samplesize: {n}")
        print(f"multithread:")
        a1=data_loader.get_stock_data_from_lst(data.loc[:n, "symbol"], multiprocess=True)
        print(f"singlethread:")
        a1=data_loader.get_stock_data_from_lst(data.loc[:n, "symbol"], multiprocess=False)
        print("")
