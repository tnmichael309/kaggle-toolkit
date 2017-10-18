from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models_dict, n_folds=5, target_col = None):
        self.base_models_dict = base_models_dict
        self.n_folds = n_folds
        self.out_of_fold_predictions = pd.DataFame()
        self.test_set_predictions = pd.DataFame()
        self.target_col = 'target'
        if target_col is not None:
            self.target_col = target_col
            
    # Fit the data on clones of the original models, 
    # return generated out-of-fold predictions of training set as meta features to train on
    # and meta features of test set
    def stacked_fit(self, train_x, train_y, test_x):
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=519)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        for model_name, model in self.base_models_dict.items():
            print("\n==================\n", model_name)
            
            count = 0
            test_set_predictions = []
            out_of_fold_predictions = np.zeros((train_x.shape[0],))
            
            for train_index, holdout_index in kfold.split(train_x, train_y):
                instance = clone(model)
                instance.fit(train_x[train_index], train_y[train_index])
                
                y_pred = instance.predict(train_x[holdout_index])
                mse = mean_squared_error(train_y[holdout_index], y_pred)
                print("mse=", mse, "\n")
                
                out_of_fold_predictions[holdout_index] = y_pred
                test_set_predictions.append(instance.predict(test_x))
                
                
            self.out_of_fold_predictions[model_name] = out_of_fold_predictions
            self.test_set_predictions[model_name] = np.column_stack(test_set_predictions).mean(axis=1)
        
        self.out_of_fold_predictions[self.target_col] = train_y
        
        # return the dataframe for meta features to train and test
        return self.out_of_fold_predictions, self.test_set_predictions
    
    def save_to_csv(self, prefix):
        self.out_of_fold_predictions.to_csv(prefix + '_meta_train.csv', encoding = 'utf-8', index=False)
        self.test_set_predictions.to_csv(prefix + '_meta_test.csv', encoding = 'utf-8', index=False)
        
    # Train the meta model on the out-of-fold predictions and predict the final result
    def meta_fit(self, meta_model_):
        instance = clone(meta_model_)
        features = list(self.out_of_fold_predictions.columns)
        features.remove(self.target_col)
        
        x = self.out_of_fold_predictions[features]
        y = self.out_of_fold_predictions[self.target_col]
        instance.fit(x, y)
        
        pred_y = instance.predict(x)
        print("meta model's training set mse= ", mean_squared_error(pred_y, y), "\n")
        
        return instance.predict(self.test_set_predictions)
        
    