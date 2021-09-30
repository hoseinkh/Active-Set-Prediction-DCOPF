# Here we use Logistic regression for classification! Here, each congestion pattern is classified as a hole.
#
#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import os
#
#
#
use_cross_validation = True
# if this variable is set to <True>, the code uses cross validation. If this variable ...
# ... is set to <False>, then it applies the LDA algorithm once, and there is no cross validation
#
#
## Step 1: Reading the data of scenarios! These data are stored in the directory named <Data>, and are of .csv format!
#
curr_dir = os.getcwd()
dir_of_saved_ndarray_files = curr_dir + '/Data/'
#
# matrix of all feasible samples!
df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_feasible_samples.csv', sep=',',header=None)
All_feasible_samples = df.values
#
# matrix of actual congestion patterns corresponding to the feasible samples!
df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_congestions_patterns___feasible.csv', sep=',',header=None)
All_matrix_of_congestion_patterns_feasible = df.values
#
# vector of single number labels!
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_feasible_samples___single_number_labeling.csv', sep=',',header=None)
All_labels_feasible_vector_of___single_number_labeling = df.values
#
# matrix of basis vector labels!
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_feasible_samples___basis_vector_labeling.csv', sep=',',header=None)
All_labels_feasible_matrix_of___basis_vector_labeling = df.values
#
# matrix of distictive congestion patterns!
df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_distinctive_cong_pattern.csv', sep=',',header=None)
matrix_of_distinctive_cong_pattern = df.values
#
# single-number labels corresponding to the distinctive congestion patterns in <matrix_of_distinctive_cong_pattern>
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_dist_feasible_samples___single_number_labeling.csv', sep=',',header=None)
labels_of_distinctive_congestion_patterns___single_number_labeling = df.values
#
# # basis-vector labels corresponding to the distinctive congestion patterns in <matrix_of_distinctive_cong_pattern>
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_dist_feasible_samples___basis_vector_labeling.csv', sep=',',header=None)
labels_of_distinctive_congestion_patterns___basis_vector_labeling = df.values
#
# array of number of samples that has the same congestion pattern as each distinctive congestion patterns in <matrix_of_distinctive_cong_pattern>
df = pd.read_csv(dir_of_saved_ndarray_files + 'array_of_num_of_patterns_for_each_dist_cong_pttrn.csv', sep=',',header=None)
array_of_num_of_patterns_for_each_dist_cong_pttrn = df.values
#
# matrix of all infeasible samples!
df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_infeasible_samples.csv', sep=',',header=None)
All_infeasible_samples = df.values
#
#
num_of_samples_with_feasible_solution , num_of_RPPs = All_feasible_samples.shape
_ , num_of_distinctive_classes = All_labels_feasible_matrix_of___basis_vector_labeling.shape
num_of_branches = len(All_matrix_of_congestion_patterns_feasible[0])
#
#
sample_and_label_data = np.concatenate((All_feasible_samples, All_labels_feasible_matrix_of___basis_vector_labeling, All_labels_feasible_vector_of___single_number_labeling), axis = 1)
np.random.shuffle(sample_and_label_data) # Shuffles the rows
All_feasible_samples___after_shuffle                                       = sample_and_label_data[:,0:num_of_RPPs]
All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle      = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_distinctive_classes]
All_labels_feasible_vector_of___single_number_labeling___after_shuffle     = sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:].reshape(len(sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:]),)
#
#
#
if use_cross_validation:
    #
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = LogisticRegression(multi_class='auto', solver='liblinear')
    scoring = 'accuracy'
    results = model_selection.cross_val_score(model, All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, cv=kfold, scoring=scoring)
    print("Accuracy: {}% (+/- {}%)".format(results.mean() * 100, results.std() * 100))
    5+6
    #
else: # no cross validation!
    # Split data to train and test on 80-20 ratio
    X_train, X_test, y_train, y_test = train_test_split(All_feasible_samples___after_shuffle,
                                                        All_labels_feasible_vector_of___single_number_labeling___after_shuffle,
                                                        test_size=0.2, random_state=0)
    #
    # Create logistic regression model
    logreg = LogisticRegression(random_state=0)
    #
    # Train classifier
    logreg.fit(X_train, y_train)
    #
    # Make predictions on unseen test data
    y_pred = logreg.predict(X_test)
    #
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix : \n", cm) # for more info on the confusion matrix see the following link:
    # ... https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
    #
    forecasting_accuracy = (np.trace(cm) / np.sum(cm)) * 100
    print("forecasting_accuracy = {}%".format(forecasting_accuracy))
    #
    forecasting_error = ((np.sum(cm) - np.trace(cm)) / np.sum(cm)) * 100
    print("forecasting_error = {}%".format(forecasting_error))
    5 + 6
#
#
#
