import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
        
class RandomForestModel(Model):
    def train(self, X_train: pd.DataFrame, Y_train: pd.Series, **kwargs):
        try:
            clf = RandomForestClassifier(**kwargs)
            clf.fit(X_train, Y_train)
            logging.info("Random Forest model training completed")
            return clf
        except Exception as e:
            logging.error(f"Error in training the Random Forest model: {e}")
            raise e

class SVMModel(Model):
    def train(self, X_train: pd.DataFrame, Y_train: pd.Series, **kwargs):
        try:
            clf = SVC(**kwargs)
            clf.fit(X_train, Y_train)
            logging.info("SVM model training completed")
            return clf
        except Exception as e:
            logging.error(f"Error in training the SVM model: {e}")
            raise e        
