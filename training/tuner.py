from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import r2_score
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from xgboost import XGBRegressor

class Model_Finder:

    def __init__(self):
        self.clf = RandomForestClassifier()
        self.DecisionTreeReg = DecisionTreeRegressor()


    def get_best_params_for_random_forest(self,train_x,train_y):
        try:
            # initializing with different combination of parameters
            param_grid = {"n_estimators": [10, 50, 100, 130], 
                               "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), 
                               "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=self.clf, param_grid=param_grid, cv=5,  verbose=3)
            #finding the best parameters
            grid.fit(train_x, train_y)

            #extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            n_estimators = grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                              max_depth=max_depth, max_features=max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            return self.clf
            
        except Exception as e:
            print("Exception occured in getting best parameters for random forest")
            raise Exception()

    def get_best_params_for_DecisionTreeRegressor(self, train_x, train_y):
        try:
            # initializing with different combination of parameters
            param_grid_decisionTree = {"criterion": ["mse", "friedman_mse", "mae"],
                              "splitter": ["best", "random"],
                              "max_features": ["auto", "sqrt", "log2"],
                              'max_depth': range(2, 16, 2),
                              'min_samples_split': range(2, 16, 2)
                              }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(self.DecisionTreeReg, param_grid_decisionTree, verbose=3,cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_['criterion']
            splitter = grid.best_params_['splitter']
            max_features = grid.best_params_['max_features']
            max_depth  = grid.best_params_['max_depth']
            min_samples_split = grid.best_params_['min_samples_split']

            # creating a new model with the best parameters
            self.decisionTreeReg = DecisionTreeRegressor(criterion=criterion,splitter=splitter,max_features=max_features,max_depth=max_depth,min_samples_split=min_samples_split)
            # training the mew models
            self.decisionTreeReg.fit(train_x, train_y)
            return self.decisionTreeReg
        except Exception as e:
            print("Exception occured in getting best parameters for decision tree regressor")
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):
        try:
            # initializing with different combination of parameters
            param_grid_xgboost = {
                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]
            }
            # Creating an object of the Grid Search class
            grid= GridSearchCV(XGBRegressor(objective='reg:linear'),param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            learning_rate = grid.best_params_['learning_rate']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBRegressor(objective='reg:linear',learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            return self.xgb

        except Exception as e:
            print("Exception occured in getting best parameters for xgboost")
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
        try:
            self.decisionTreeReg= self.get_best_params_for_DecisionTreeRegressor(train_x, train_y)
            prediction_decisionTreeReg = self.decisionTreeReg.predict(test_x)
            decisionTreeReg_error = r2_score(test_y,prediction_decisionTreeReg)

            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            prediction_xgboost = self.xgboost.predict(test_x) 
            prediction_xgboost_error = r2_score(test_y,prediction_xgboost)


            #comparing the two models
            if(decisionTreeReg_error <  prediction_xgboost_error):
                return 'XGBoost',self.xgboost
            else:
                return 'DecisionTreeReg',self.decisionTreeReg

        except Exception as e:
            print("Exception occured in finding best model")
            raise Exception()

