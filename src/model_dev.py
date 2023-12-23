
import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class Model(ABC):
    @abstractmethod
    def train(self, X_train:pd.DataFrame, Y_train:pd.Series):
        pass    

class LogisticRegressionModel(Model):
    def train(self, X_train:pd.DataFrame, Y_train:pd.Series, **kwargs):
        try:
            clf = LogisticRegression(**kwargs)
            clf.fit(X_train, Y_train)
            logging.info("Model training completed")
            return clf 
        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            raise e


    class DecisionTreeClassifierModel(Model):
        """
        DecisionTreeClassifierModel that implements the Model interface.
        """

        def train(self, x_train, y_train, **kwargs):
            try:
                clf = DecisionTreeClassifier(**kwargs)
                clf.fit(x_train, y_train)
                return clf
            except Exception as e:
                logging.error(f"Error in training the model: {e}") 
                raise e
        
