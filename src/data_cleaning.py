import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessing data
        """
        try:
            data["race"].fillna(data["race"].mode(),inplace=True)
            data["charges"].fillna(data["charges"].median(),inplace=True)
            data["totcst"].fillna(data["totcst"].median(),inplace=True)
            data["avtisst"].fillna(data["avtisst"].median(),inplace=True)
            data["wblc"].fillna(data["wblc"].median(),inplace=True)
            data["crea"].fillna(data["crea"].median(),inplace=True) 

            # Set the threshold for null values (e.g., 15%)
            null_threshold = 0.15

            # Calculate the percentage of null values for each column
            null_percentages = data.isnull().mean()

            # Filter columns based on the null threshold
            columns_to_drop = null_percentages[null_percentages > null_threshold].index.tolist()

            # Drop selected columns and return the modified DataFrame
            data = data.drop(columns=columns_to_drop) 
            
            data = data.drop(
                [
                    "hospdead"
                   
                ],axis=1
            )
            data=data.dropna()
            le = LabelEncoder()
            data['sex'] = le.fit_transform(data['sex'])
            data['dzgroup'] = le.fit_transform(data['dzgroup'])
            data['dzclass'] = le.fit_transform(data['dzclass'])
            data['race'] = le.fit_transform(data['race'])
            return data
    
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        try:
            X = data.drop(["death"],axis=1)
            y = data["death"]
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)          
            
            return X_train, X_test, y_train, y_test 
        
        except Exception as e:
            logging.error("Error in dividing data:{}".format(e))
            raise e

class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into train and test
    """
    def __init__(self,data: pd.DataFrame,Strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = Strategy
        
    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        """
        Handled data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e