from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class calculate:
    def __init__(self):
        pass
    
    def remove_columns(self,data,columns):
        try:
            data= data.drop(columns,axis=1)
            return data
        except Exception as e:
            print("Exception occured in remove_columns method")
            return data

    def replaceInvalidValueswithNull(self,data):
        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data

    def is_null_present(self,data):
        for i in data.columns:
            count=data[i].isnull().sum()
            if count>0:
                return False
        return True 

    def impute_missing_values(self,data):
        try:
            imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            s=imputer.fit_transform(data) # impute the missing values
            new_data=pd.DataFrame(data=(s), columns=data.columns)
            return new_data
        except Exception as e:
            print("Exception occured in impute_missing_values method")

    
    def separate_label_feature(self,data,label_column_name):
        try:
            X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            Y=data[label_column_name] # Filter the Label columns
            return X,Y
        except Exception as e:
            print("Exception occured in separate_label_feature method")

    def standardScalingData(self,X):
        scalar = StandardScaler()
        X_scaled = scalar.fit_transform(X)
        return X_scaled
