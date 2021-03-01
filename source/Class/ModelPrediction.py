import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
from .TrainingData import EvaluationData


class ModelPrediction(RandomForestRegressor):
    
    def __init__(self, ed: EvaluationData):
        
        RandomForestRegressor.__init__(self, n_estimators = 1000, random_state = 42)# Train the model on training data
        self.fit(ed.X_train, ed.y_train);    
        self.t = ed.t
    
        
    def predictSeries(self, series: pd.Series, normalize: bool = True) -> float:
        
        X      = series.values.reshape(1, -1)
        y_pred = self.predict(X)[0]
                
        return self.scoreMapper(y_pred) if normalize else y_pred
    
    
    def scoreMapper(self, x: float) -> float:
        
        score = (x / self.t) ** 0.6 * 10.5
        
        return min(score, 10)
    
    def toPickle(self, path: str):

        pickle.dump(self, open(path, 'wb'))
        
        
    def readPickle(path: str):
        return pickle.load(open(path, 'rb'))
        