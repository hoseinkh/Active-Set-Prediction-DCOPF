# This file uses LDA to classify congestions. Each congestion pattern is classified as a hole.
#
import numpy as np
import pandas as pd
#
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#
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
#
sample_and_label_data = np.concatenate((All_feasible_samples, All_labels_feasible_matrix_of___basis_vector_labeling, All_labels_feasible_vector_of___single_number_labeling), axis = 1)
np.random.shuffle(sample_and_label_data) # Shuffles the rows
All_feasible_samples___after_shuffle                                       = sample_and_label_data[:,0:num_of_RPPs]
All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle      = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_distinctive_classes]
All_labels_feasible_vector_of___single_number_labeling___after_shuffle     = sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:].reshape(len(sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:]),)
#
#
#
#
if use_cross_validation:  # use cross validation: apply LDA multiple times
    # # Make predictions on unseen test data  +  Cross-validation error checking
    scores = cross_val_score(LDA(), All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, cv=10)
    print("Accuracy: {}% (+/- {}%)".format(scores.mean() * 100, (scores.std() * 100) * 2))
else:
    # # Split data to train and test on 80-20 ratio
    X_train, X_test, y_train, y_test = train_test_split(All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, test_size=0.2, random_state=0)
    #
    # # Create LDA
    lda = LDA()
    #
    # # Train classifier
    lda.fit(X_train, y_train)
    #
    # Make predictions on unseen test data
    y_pred = lda.predict(X_test)
    print(str(lda.score(X_test, y_test) * 100) + '%')
    #
    #
    #
    # # # # Here we analyze for which samples the LDA does not work!
    # Check which congestion
    #
    misclassified = np.where(np.subtract(y_test.reshape((len(y_test)),) , lda.predict(X_test).reshape((len(y_test)),)).reshape((len(y_test)),) != 0 )
    #
    #
    list_of_frequency_of_missclassified_cong_patterns_in_testing = [0]*num_of_distinctive_classes
    list_of_frequency_of_cong_patterns_in_testing  = [0]*num_of_distinctive_classes
    list_of_frequency_of_cong_patterns_in_training  = [0]*num_of_distinctive_classes
    #
    #
    for i in misclassified[0]:
        class_number_for_current_missclassified_test_cong_pattern = int(y_test[i])  # note that we know that elements of <y_test> are always integer numbers! (because they are class numbers!)
        list_of_frequency_of_missclassified_cong_patterns_in_testing[class_number_for_current_missclassified_test_cong_pattern] += 1
    #
    for i in y_test:
        list_of_frequency_of_cong_patterns_in_testing[int(i)-1] += 1
    #
    for i in y_train:
        list_of_frequency_of_cong_patterns_in_training[int(i)-1] += 1
    #
    #
    print('Frequency of All classes:                           {}'.format(array_of_num_of_patterns_for_each_dist_cong_pttrn.tolist()))
    print('Frequency of classes in Training data:              {}'.format(list_of_frequency_of_cong_patterns_in_training))
    print('Frequency of classes in Testing data:               {}'.format(list_of_frequency_of_cong_patterns_in_testing))
    print('Frequency of missclssified classes in Testing data: {}'.format(list_of_frequency_of_missclassified_cong_patterns_in_testing))
    #

    forecasting_error = sum(list_of_frequency_of_missclassified_cong_patterns_in_testing) * 100 / sum(list_of_frequency_of_cong_patterns_in_testing)
    print("forecasting_error = {}%".format(forecasting_error))


