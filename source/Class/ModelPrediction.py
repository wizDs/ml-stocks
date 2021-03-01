from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
from .TrainingData import EvaluationData

class ModelPrediction(RandomForestRegressor):
    
    def __init__(self, ed: EvaluationData):
        
        RandomForestRegressor.__init__(self, n_estimators = 1000, random_state = 42)# Train the model on training data
        self.fit(ed.X_train, ed.y_train);    
    
        
    