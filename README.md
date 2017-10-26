# kaggle-toolkit
A kaggle toolkit development repos

This is repos for common-used functions to be used in Kaggle Competitions

stacking_models_api.py:

* class StackingAveragedModels:  
  * stacked_fit: fit with stacking model with base models, can use supervised\semi-supervised\unsupervised base model
    * fit on the train data with base models, and use the out-of-fold predictions as meta-features for the new train data
    * predict the test data: for each model, average each fold's predictions as the new test data
  * meta_fit: given the meta model, fit the meta model with new train data and predict on the new test data


* class stacking_averaged_model_tester:
  * base_cv: split data into separate train\valid set. For each set, use a StackingAveragedModels() with stacked_fit on train set, and use the meta_model to evaluate the final performance on the valid set. (Note that we could have out-of-fold prediction performance when using stacked_fit)

  * meta_cv: split data into separate train\valid set and test the meta model. Should proviede with a stacked_fitted StackingAveragedModels.

  * meta_grid_search: Do paramter grid search on the given meta mode. Should proviede with a stacked_fitted StackingAveragedModels.

tf_ann.py:
* An ann based on tensorflow to minimize rmse.
* gpu-support
* support:
  * batch normalization
  * ADAM Optimizer
  * learning rate decay
  * L1\L2 regulariztion
  * Drop out
  * Different number of fully connected hiddeen layers\neurons
  

tf_gpu_test.py:
* An easy test for gpu support with tensorflow
