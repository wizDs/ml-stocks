import numpy as np
from datetime import datetime, timedelta
from typing import List
from math import floor

from scipy.optimize import linprog
from gekko import GEKKO

from DataLoader import DataLoader


class NumberOfStocksToBuy:
    
    def __init__(self, stock_names: List[str], weights: List[float], money: int, integer_problem: bool = True):
        
        assert len(weights) == len(stock_names), 'weights and stock_names must be of same dimension'
        assert sum(weights) <= 1,                'all weights must sum to less or equal to 1'
        
        prices       = self.latestPrices(stock_names)
        self.prices  = prices        
    
        d = len(weights)
        
        
        c = np.array(prices)
        b = np.array(weights) * money
        A = np.zeros((d, d), int)
        np.fill_diagonal(A, prices) 

        if integer_problem:
            res = self.mip(c, A, b)
        else:
            res = self.lp(c, A, b)
        

        self.solution       = [int(x) for x in res]
        self.total_price    = round(np.dot(np.array(prices),  self.solution))
        
    
    def latestPrices(self, lst_symbols: List[str]) -> List[float]:
        
        ten_days_ago = datetime.today() - timedelta(days = 10)    
        dat  = DataLoader(lst_symbols, ten_days_ago).data
        
        n = 0
        while True:
            n = n + 1
        
            if all(dat.iloc[-n].notna()):
              out = dat.iloc[-n].tolist()
              break
          
        return out
    
    def mip(self, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> List[float]:
        
        m = GEKKO() # Initialize gekko
        m.options.SOLVER=1  # APOPT is an MINLP solver
        
        d = len(c)
        
        x = np.array([m.Var(value=0, integer=True) for i in range(d)])
        
        m.Obj(-np.dot(c, x))
        
        for i in range(d):
            m.Equation([np.dot(A[i], x) <= b[i]])
        
        m.solve(disp=False) # Solve
        
        return [v.value[0] for v in x]

    def lp(self, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> List[float]:
        res = linprog(-c, A_eq=A, b_eq=b, bounds = [(0, None) for i in range(len(c))])
        
        return [floor(v) for v in res.x]
    
    def __repr__(self):
        summary = """
        Total price:             {total_price}
        Prices:                  {prices}
        Number of stocks to buy: {solution}
        
        """.format(prices      = ', '.join(str(round(p, 2)) for p in self.prices),
                   total_price = self.total_price,
                   solution    = ', '.join(str(p) for p in self.solution))
        
        return summary
                 