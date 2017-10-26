from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import copy
import numpy as np
from stacking_models_api import StackingAveragedModels

def root_mean_squred_error(y1, y2):
    return np.sqrt(mean_squared_error(y1, y2))
    
    
def cross_validate(model, X, y, fold_num, error_func=mean_squared_error):
    kf = KFold(n_splits=fold_num, shuffle=True, random_state=519)
    counter  = 0
    mean_err = 0
    
    for train_index, valid_index in kf.split(X,y):
        instance = copy.deepcopy(model)    
        instance.fit(X.loc[train_index], y[train_index])
        y_pred = instance.predict(X.loc[valid_index])
        counter += 1
        err = error_func(y_pred, y[valid_index])
        mean_err += err
        print("fold ", counter, " valid error: ", err)
        
    print(fold_num, " fold(s) avg. valid error: ", mean_err / fold_num)
    