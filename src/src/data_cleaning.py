import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
#here, abc refers to the module and ABC refers to the Abstract Base Classes
from typing import Union
# Import OneHotEncoder from sklearn
from sklearn.preprocessing import OneHotEncoder

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    
    """This class is used to preprocess the given dataset"""
    def handle_data(self, data: pd.DataFrame) ->Union[pd.DataFrame,pd.Series]:
        try:
            print("Column Names Before Preprocessing:", data.columns)  
            data = data.drop([ 'hospdead','edu', 'income', 'totmcst', 'pafi', 'alb', 'bili', 'ph', 'glucose', 'bun', 'urine', 'adlp', 'adls', 'sfdm2'], axis=1)
            print(data.columns)
            #print(data.columns())
            # if 'death' in data.columns:
            #     print("Death column found in data.")
            # else:
            #     print("Death column not found in data.")
            # data["Attrition"] = data["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
            # data["Over18"] = data["Over18"].apply(lambda x: 1 if x == "Yes" else 0)
            # data["OverTime"] = data["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)
            # Set the threshold for null values (e.g., 15%)
            # null_threshold = 0.15

            # # Calculate the percentage of null values for each column
            # null_percentages = data.isnull().mean()

            # # Filter columns based on the null threshold
            # columns_to_drop = null_percentages[null_percentages > null_threshold].index.tolist()
            # print("Columns-",columns_to_drop)

            # Drop selected columns and return the modified DataFrame
            #data = data.drop(columns=columns_to_drop)   
            data["race"].fillna(data["race"].mode(),inplace=True)
            data["charges"].fillna(data["charges"].median(),inplace=True)
            data["totcst"].fillna(data["totcst"].median(),inplace=True)
            data["avtisst"].fillna(data["avtisst"].median(),inplace=True)
            data["wblc"].fillna(data["wblc"].median(),inplace=True)
            data["crea"].fillna(data["crea"].median(),inplace=True)   

            # Extract categorical variables
            cat = data[['sex', 'dzgroup', 'dzclass', 'race']]

            # Perform one-hot encoding on categorical variables
            onehot = OneHotEncoder()
            cat_encoded = onehot.fit_transform(cat).toarray()
            #to have the feature unqiue data as their respective encoded column names
            feature_names = onehot.get_feature_names_out(input_features=cat.columns)
            # Convert cat_encoded to DataFrame
            cat_df = pd.DataFrame(cat_encoded,columns=feature_names)
            print(cat_df.head())
            # Extract numerical variables
            numerical = data[['age', 'death', 'slos', 'd.time', 'num.co', 'scoma','charges', 'totcst', 'avtisst', 'meanbp', 'wblc', 'hrt', 'resp','temp', 'crea', 'sod', 'adlsc']]

            # Concatenate X_cat_df and X_numerical
            data = pd.concat([cat_df, numerical], axis=1)

            print("Column Names After Preprocessing:", data.columns)  # Add this line
            print("Preprocessed Data:")
            print(data.head())
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e



class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # Check if 'Attrition' is present in the data
            if 'death' in data.columns:
                X = data.drop(['death'], axis=1)
                Y = data['death']
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                return X_train, X_test, Y_train, Y_test
            else:
                raise ValueError("Death column not found in data.")
        except Exception as e:
            logging.error(f"Error in data handling: {str(e)}")
            raise e

class DataCleaning:
            def __init__(self,data:pd.DataFrame,strategy:DataStrategy)->None:
                self.data=data
                self.strategy=strategy
            def handle_data(self)->Union[pd.DataFrame,pd.Series]:
                try:
                    return self.strategy.handle_data(self.data)
                except Exception as e:
                    logging.error(f"There is a error in dataHandling{e}")
                    raise e
                        
                 

              