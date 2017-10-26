from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV

class stacking_averaged_model_tester(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self, n_folds=5, target_col = None, random_state = 0):
        self.n_folds = n_folds
        self.target_col = 'target'
        if target_col is not None:
            self.target_col = target_col
        self.random_state = random_state
        
    def base_cv(train, 
        sl_base_models_dict = None, 
        semi_sl_base_models_dict = None, 
        usl_base_models_dict = None,
        meta_model = ElasticNetCV(cv=5, random_state=0, max_iter=10000, l1_ratio=1.0)):
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        if self.target_col not in train:
            raise ValueError(target_col, "must in given train data")
            
        train_x = train[list(train.columns).remove(self.target_col)]
        train_y = train[self.target_col]
        avg_valid_rmse = 0.0
        
        for train_index, holdout_index in kfold.split(train_x, train_y):
            sam = StackingAveragedModels(n_folds = self.n_folds, target_col=self.target_col)
        
            used_train_x = train_x.loc[train_index]
            used_train_y = train_y[train_index]
            used_test_x = train_x.loc[holdout_index]
            used_test_y = train_y[holdout_index]
            
            # using stacked_fit will have insight about
            # the out_of_fold_predictions' validation errors
            if sl_base_models_dict not None:
                sam.stacked_fit(sl_base_models_dict, 
                    used_train_x, used_test_x, used_train_y, random_state=self.random_state, 
                    method='supervised')
            
            if semi_sl_base_models_dict not None:
                sam.stacked_fit(semi_sl_base_models_dict, 
                    used_train_x, used_test_x, used_train_y, random_state=self.random_state, 
                    method='semi-supervised')        
            
            if usl_base_models_dict not None:
                sam.stacked_fit(usl_base_models_dict, 
                    used_train_x, used_test_x, used_train_y, random_state=self.random_state, 
                    method='unsupervised')
            
            # finally use meta fit to fit on meta features (out-of-fold predictions) and predict
            # the test data with meta-features 
            # we can then have insight about test set rmse
            if sl_base_models_dict not None or semi_sl_base_models_dict not None or usl_base_models_dict not None:
                predictions = sam.meta_fit(meta_model)
                rmse = np.sqrt(mean_squared_error(predictions, used_test_y))
                avg_valid_rmse += rmse
                print("Test rmse=", rmse)
            else:
                pass
            
        print('Test avg. rmse =', avg_valid_rmse/self.n_folds)
    
    # need to provide with trained sam
    def meta_grid_search(self, sam, meta_model, meta_model_cv_params=None):
        train, _ = sam.get_meta_train_test_dataframe()
        
        if self.target_col not in train:
            raise ValueError(self.target_col, "must in the trained stacking averaged model")
            
        train_x = train[list(train.columns).remove(self.target_col)]
        train_y = train[self.target_col]
        
        gs = GridSearchCV(estimator=meta_model, param_grid = meta_model_cv_params, scoring='neg_mean_squared_error', iid=False, cv=5)
        gs.fit(train_x, train_y)
        print(gs.best_params_, np.sqrt(gs.best_score_*-1))

    # need to provide with trained sam
    def meta_cv(self, sam, meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        train, _ = sam.get_meta_train_test_dataframe()
        
        if self.target_col not in train:
            raise ValueError(self.target_col, "must in the trained stacking averaged model")
            
        train_x = train[list(train.columns).remove(self.target_col)]
        train_y = train[self.target_col]
        
        avg_train_rmse = 0.0
        avg_valid_rmse = 0.0
        
        for train_index, holdout_index in kfold.split(train_x, train_y):
            instance = copy.deepcopy(meta_model)
            
            used_train_x = train_x.loc[train_index]
            used_train_y = train_y[train_index]
            used_test_x = train_x.loc[holdout_index]
            used_test_y = train_y[holdout_index]
            
            instance.fit(used_train_x, used_train_y)
            pred_train_y = instance.predict(used_train_x)
            pred_test_y = instance.predict(used_test_x)
            
            train_rmse = np.sqrt(mean_squared_error(pred_train_y, used_train_y))
            valid_rmse = np.sqrt(mean_squared_error(pred_test_y, used_test_y))
            
            print("Train rmse=", train_rmse)
            print("Valid rmse=", valid_rmse)
            
            avg_train_rmse += train_rmse
            avg_valid_rmse += valid_rmse
            
        print("Train avg rmse=", avg_train_rmse/self.n_folds)
        print("Valid avg rmse=", avg_valid_rmse/self.n_folds)

        
class StackingAveragedModels():
    def __init__(self, n_folds=5, target_col = None):
        self.n_folds = n_folds
        self.out_of_fold_predictions = pd.DataFrame()
        self.test_set_predictions = pd.DataFrame()
        self.target_col = 'target'
        if target_col is not None:
            self.target_col = target_col
    
        self.fitter_dict = {
            'supervised': self._supervised_stacked_fit,
            'semi-supervised': self._semi_supervised_stacked_fit,
            'unsupervised': self._unsupervised_stacked_fit
        }
        
    def stacked_fit(self, base_models_dict, train_x, test_x, train_y = None, method = 'supervised', random_state = 0):
        if method not in self.fitter_dict:
            raise ValueError('Invalid method input: method can be only supervised, semi-supervised or unsupervised.')
        
        return self.fitter_dict[method](base_models_dict, train_x, test_x, train_y, random_state=random_state)
        
    # Fit the data on clones of the original models, 
    # return generated out-of-fold predictions of training set as meta features to train on
    # and meta features of test set
    # For supervised learning with fit predict methods
    def _supervised_stacked_fit(self, base_models_dict, train_x, test_x, train_y, random_state):
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=random_state)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        for model_name, model in base_models_dict.items():
            print("\n==================\n", model_name)
            
            rmse_sum = .0
            test_set_predictions = []
            out_of_fold_predictions = np.zeros((train_x.shape[0],))
            
            for train_index, holdout_index in kfold.split(train_x, train_y):
                instance = copy.deepcopy(model)
                instance.fit(train_x.loc[train_index], train_y[train_index])
                
                y_pred = instance.predict(train_x.loc[holdout_index])
                rmse = np.sqrt(mean_squared_error(train_y[holdout_index], y_pred))
                rmse_sum += rmse
                print("rmse=", rmse)
                
                out_of_fold_predictions[holdout_index] = y_pred
                test_set_predictions.append(instance.predict(test_x))
                
            print("Avg rmse = ", rmse_sum/self.n_folds)
            
            self.out_of_fold_predictions[model_name] = out_of_fold_predictions
            self.test_set_predictions[model_name] = np.column_stack(test_set_predictions).mean(axis=1)
        
        self.out_of_fold_predictions[self.target_col] = train_y
        
        # return the dataframe for meta features to train and test
        return self.get_meta_train_test_dataframe()
    
    # for semi supervised methods: knn regressor with fit, kneighbors and predict methods
    # it used the whole dataset instead of out-of-fold predictions
    # output features: knn regressor's result + distances to n-nearest neighbors
    def _semi_supervised_stacked_fit(self, base_models_dict, train_x, test_x, train_y, random_state):
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=random_state)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        for model_name, model in base_models_dict.items():
            print("\n==================\n", model_name)
            
            rmse_sum = .0
            test_set_predictions = []
            test_set_predictions_dist = []
            out_of_fold_predictions = np.zeros((train_x.shape[0],))
            out_of_fold_predictions_dist = np.zeros((train_x.shape[0],))
            
            for train_index, holdout_index in kfold.split(train_x, train_y):
                instance = copy.deepcopy(model)
                instance.fit(train_x.loc[train_index], train_y[train_index])
                
                y_pred = instance.predict(train_x.loc[holdout_index])    
                distances, _ = instance.kneighbors(train_x.loc[holdout_index]) # get n nearest neighbors distances
                out_of_fold_predictions[holdout_index] = y_pred
                out_of_fold_predictions_dist[holdout_index] = np.array(distances).mean(axis=1)
                
                rmse = np.sqrt(mean_squared_error(train_y[holdout_index], y_pred))
                rmse_sum += rmse
                print("rmse=", rmse)
                
                y_pred_test = instance.predict(test_x)
                test_distances, _ = instance.kneighbors(test_x) # get n nearest neighbors distances
                test_set_predictions.append(y_pred_test)
                test_set_predictions_dist.append(np.array(test_distances).mean(axis=1))
                
            print("Avg rmse = ", rmse_sum/self.n_folds)
            
            self.out_of_fold_predictions[model_name] = out_of_fold_predictions
            self.out_of_fold_predictions[model_name + '_dist'] = out_of_fold_predictions_dist
            self.test_set_predictions[model_name] = np.column_stack(test_set_predictions).mean(axis=1)
            self.test_set_predictions[model_name + '_dist'] = np.column_stack(test_set_predictions_dist).mean(axis=1)
        
        self.out_of_fold_predictions[self.target_col] = train_y
        
        # return the dataframe for meta features to train and test
        return self.get_meta_train_test_dataframe()
        
    # for unsupervised methods: clustering methods with fit and predict methods
    # it used the whole dataset instead of out-of-fold predictions
    # dummies (one-hot encodings) for clustering labels
    def _unsupervised_stacked_fit(self, base_models_dict, train_x, test_x, train_y, random_state):
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        for model_name, model in base_models_dict.items():
            print("\n==================\n", model_name)
            
            instance = copy.deepcopy(model)
            instance.fit(train_x)
                
            y_pred = instance.predict(train_x)    
            self.out_of_fold_predictions[model_name] = y_pred
            
            y_pred_test = instance.predict(test_x)
            self.test_set_predictions[model_name] = y_pred_test    
            
        model_names = [model_name for model_name, model in base_models_dict.items()]
        
        # concat together and get dummies
        # if we get dummies separately, we might come up with different number of dummy columns
        length = self.out_of_fold_predictions.shape[0]
        all_df = pd.concat([self.out_of_fold_predictions, self.test_set_predictions], join="inner")
        all_df = pd.get_dummies(all_df, columns=model_names)
        self.out_of_fold_predictions = all_df[:length]
        self.test_set_predictions = all_df[length:]
        
        #self.out_of_fold_predictions = pd.get_dummies(self.out_of_fold_predictions, columns=model_names)
        #self.test_set_predictions = pd.get_dummies(self.test_set_predictions, columns=model_names)
        
        self.out_of_fold_predictions[self.target_col] = train_y
            
        # return the dataframe for meta features to train and test
        return self.get_meta_train_test_dataframe()
    
    def get_meta_train_test_dataframe(self):
        return self.out_of_fold_predictions, self.test_set_predictions
        
    def save_to_csv(self, prefix):
        self.out_of_fold_predictions.to_csv(prefix + '_meta_train.csv', float_format='%.6f', encoding = 'utf-8', index=False)
        self.test_set_predictions.to_csv(prefix + '_meta_test.csv', float_format='%.6f', encoding = 'utf-8', index=False)
        
    # Train the meta model on the out-of-fold predictions and predict the final result
    def meta_fit(self, meta_model):
        instance = copy.deepcopy(meta_model)
        features = list(self.out_of_fold_predictions.columns)
        features.remove(self.target_col)
        
        x = self.out_of_fold_predictions[features]
        y = self.out_of_fold_predictions[self.target_col]
        instance.fit(x, y)
        
        pred_y = instance.predict(x)
        print("meta model's training set mse= ", np.sqrt(mean_squared_error(pred_y, y)), "\n")
        
        return instance.predict(self.test_set_predictions)
    