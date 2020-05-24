# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:46:49 2020

@author: Wilson
"""

import numpy as np
import pandas as pd
import time
from time import sleep
import requests

from typing import List
from data_loader import data_loader, multithread

# =============================================================================
# 
# =============================================================================
url = 'https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download'
stock_tickers = pd.read_csv(url).rename(str.lower, axis = "columns")
test_tickers = stock_tickers.loc[:100, "symbol"]

# =============================================================================
# 
# =============================================================================
#a1=data_loader.get_stock_data_from_lst(test_tickers, combine =False)
a2=data_loader.get_stock_data_from_lst(stock_tickers.loc[:, "symbol"], combine= False)



#hastighed
performance_test=[
    (60,   39.73038935661316,  38.41385245323181),
    (100,  70.74625849723816,  88.26604461669922),
    (150, 116.33711552619934,  136.13779997825623),
    (300, 262.5901942253113, 263.67331051826477),
    (500, 440.2024414539337)
]
(3528.2286722660065)
