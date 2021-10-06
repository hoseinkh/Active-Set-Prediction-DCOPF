# This file uses SVM to classify each line and see whether that line is congested or not! Then it put these ...
# ... predictions of the lines together to make a prediction about the congestion pattern (as a whole!)
#
import numpy as np
import pandas as pd
#
from sklearn import svm
#
import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
#
#
use_cross_validation = True
# if this variable is set to <True>, the code uses cross validation. If this variable ...
# ... is set to <False>, then it applies the LDA algorithm once, and there is no cross validation
#
#
## Step 1: Reading the data of scenarios! These data are stored in the directory named <Data>, and are of .csv format!
#
if True:
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
    try:
        df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_infeasible_samples.csv', sep=',', header=None)
        All_infeasible_samples = df.values
    except pd.errors.EmptyDataError: # the matrix of infeasible cases is empty!
        pass

    #
    #
    num_of_samples_with_feasible_solution , num_of_RPPs = All_feasible_samples.shape
    _ , num_of_distinctive_classes = All_labels_feasible_matrix_of___basis_vector_labeling.shape
    num_of_branches = len(All_matrix_of_congestion_patterns_feasible[0])
    #
    #
    sample_and_label_data = np.concatenate((All_feasible_samples, All_matrix_of_congestion_patterns_feasible, All_labels_feasible_matrix_of___basis_vector_labeling, All_labels_feasible_vector_of___single_number_labeling), axis = 1)
    np.random.shuffle(sample_and_label_data) # Shuffles the rows
    All_feasible_samples___after_shuffle                                       = sample_and_label_data[:,0:num_of_RPPs]
    All_matrix_of_congestion_patterns_feasible___after_shuffle                 = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_branches]
    All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle      = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_branches + num_of_distinctive_classes]
    All_labels_feasible_vector_of___single_number_labeling___after_shuffle     = sample_and_label_data[:,num_of_RPPs + num_of_branches + num_of_distinctive_classes:].reshape(len(sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:]),)
    #
    #
    #
    # if use_cross_validation:
    #
    #
    # Split data to train and test on 80-20 ratio
    X_train, X_test, y_train, y_test = train_test_split(All_feasible_samples___after_shuffle, All_matrix_of_congestion_patterns_feasible___after_shuffle, test_size = 0.2, random_state=0)
    #
    #
elif False:
    curr_dir = os.getcwd()
    dir_of_saved_ndarray_files = curr_dir + '/Data/'
    #
    X_train_80_percent = pd.read_csv(curr_dir + '/PaperResults/' + '/80PercentTraining/' + 'train_data_80.csv', sep=',', header=None)
    X_train_80_percent = X_train_80_percent.values
    #
    y_train_80_percent = pd.read_csv(curr_dir + '/PaperResults/' + '/80PercentTraining/' + 'train_label_80.csv', sep=',',header=None)
    y_train_80_percent = y_train_80_percent.values
    #
    X_test_80_percent = pd.read_csv(curr_dir + '/PaperResults/' + '/80PercentTraining/' + 'test_data_80.csv', sep=',',header=None)
    X_test_80_percent = X_test_80_percent.values
    #
    y_test_80_percent = pd.read_csv(curr_dir + '/PaperResults/' + '/80PercentTraining/' + 'test_label_80.csv', sep=',',header=None)
    y_test_80_percent = y_test_80_percent.values
    #
    #
    ratio_of_80_percent_that_we_are_going_to_use = 0.01;
    X_train , _ , y_train, _ = train_test_split(X_train_80_percent, y_train_80_percent, test_size=((0.8 - ratio_of_80_percent_that_we_are_going_to_use)/0.8), random_state=0)
    X_test = X_test_80_percent
    y_test = y_test_80_percent
    #
    num_of_samples_with_feasible_solution, num_of_RPPs = X_train_80_percent.shape
    # _, num_of_distinctive_classes = All_labels_feasible_matrix_of___basis_vector_labeling.shape
    num_of_branches = y_train.shape[1]
    #
    #
    # sample_and_label_data = np.concatenate((All_feasible_samples, All_matrix_of_congestion_patterns_feasible,All_labels_feasible_matrix_of___basis_vector_labeling,All_labels_feasible_vector_of___single_number_labeling), axis=1)
    # np.random.shuffle(sample_and_label_data)  # Shuffles the rows
    # All_feasible_samples___after_shuffle = sample_and_label_data[:, 0:num_of_RPPs]
    # All_matrix_of_congestion_patterns_feasible___after_shuffle = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_branches]
    # All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_branches + num_of_distinctive_classes]
    # All_labels_feasible_vector_of___single_number_labeling___after_shuffle = sample_and_label_data[:,num_of_RPPs + num_of_branches + num_of_distinctive_classes:].reshape(len(sample_and_label_data[:, num_of_RPPs + num_of_distinctive_classes:]), )
    #
    #
else:
    pass

#
#
## Here we set the training data for each line and classifier using a dictionary:
num_of_train_samples , _ = X_train.shape
num_of_test_samples , _ = X_test.shape
#
y_train_ALL_dictionary = {}
y_test_ALL_dictionary = {}
y_predict_ALL_dictionary = {}
#
for i in range(0, num_of_branches):
    y_train_ALL_dictionary['X_train___Line_' + str(int(i))] =  y_train[:,int(i)]
#
for i in range(num_of_branches):
    y_test_ALL_dictionary['X_est___Line_' + str(int(i))] =  y_test[:,int(i)]
#
#
## Define classifers for each line:
Classifiers_ALL_dictionary = {}
for i in range(0,num_of_branches):
    # Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))] = svm.SVC(kernel='linear', C=1)
    Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))] = svm.SVC(cache_size=200, class_weight=None, coef0=0.0, degree=3,
    gamma=0.0007, kernel='rbf', max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.001, verbose=False)
#
#
## Train the classifiers:
list_of_classes_with_only_one_type_of_label_in_the_training_data = []
for i in range(0,num_of_branches):
    print("Percentage of Training = {}%".format(i*100/num_of_branches))
    # check to see if there is more than one sample in the training labels
    if len(np.unique(y_train_ALL_dictionary['X_train___Line_' + str(int(i))]).tolist()) > 1:
        # this means that in the training data, this line could have different congestion patterns
        Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))].fit(X_train , y_train_ALL_dictionary['X_train___Line_' + str(int(i))])
    else: # there is only one class in the
        # this means that in the training data, this line has only one type of congestion!
        list_of_classes_with_only_one_type_of_label_in_the_training_data.append(int(i))
#
#
#
# Make predictions on unseen test data
for i in range(0,num_of_branches):
    if int(i) in list_of_classes_with_only_one_type_of_label_in_the_training_data:
        # because in the training data we have only one type of label (congestion) for this line, in the prediction we choose the same label!
        # so no need for SVM! (This is actually SVM, but SVM algorithm implemented in Python requires at least two different of labels in the training set!)
        y_predict_ALL_dictionary['predictions___for_Line_' + str(int(i))] = np.array([ np.unique(y_train_ALL_dictionary['X_train___Line_' + str(int(i))]).tolist()[0] ] * num_of_test_samples).reshape((num_of_test_samples,))
        #
        # number_of_missclassifies = np.count_nonzero(y_predict_ALL_dictionary['predictions___for_Line_' + str(int(0))] - y_test_ALL_dictionary['X_est___Line_' + str(int(0))])
        number_of_missclassifies = np.count_nonzero(y_predict_ALL_dictionary['predictions___for_Line_' + str(int(i))] - y_test_ALL_dictionary['X_est___Line_' + str(int(i))])
        print("Accuracy of predictions for line {}: {}%".format(int(i), (num_of_test_samples - number_of_missclassifies) * 100 / num_of_test_samples ))
    else:
        y_predict_ALL_dictionary['predictions___for_Line_' + str(int(i))] = Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))].predict(X_test)
        print("Accuracy of predictions for line {}: {}  --- SVM%".format(int(i) , Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))].score(X_test, y_test_ALL_dictionary['X_est___Line_' + str(int(i))]) * 100 ))
#
#
# print('\n list of lines with single class label in training data = \n {}'.format(list_of_classes_with_only_one_type_of_label_in_the_training_data))
#
#
## Construct the congestion pattern
# predicted_congestion_pattern = np.array([])
for i in range(num_of_branches):
    if i == 0:
        predicted_congestion_pattern = y_predict_ALL_dictionary['predictions___for_Line_' + str(int(i))].reshape((num_of_test_samples, 1))
    else:
        predicted_congestion_pattern = np.concatenate((predicted_congestion_pattern , y_predict_ALL_dictionary['predictions___for_Line_' + str(int(i))].reshape((num_of_test_samples,1))) , axis=1)
#
#
## Calculate the error in congestion prediction!
difference_between_prediction_and_test_label = y_test - predicted_congestion_pattern
#
num_of_misclassified_test_samples = 0
for i in range(0,num_of_test_samples):
    if len(np.nonzero(difference_between_prediction_and_test_label[i,:])[0]) != 0:
        num_of_misclassified_test_samples += 1
    else:
        pass
#
#
#
print('Total Forecasting Accuracy = {}%'.format((num_of_test_samples - num_of_misclassified_test_samples) * 100 / num_of_test_samples))
#
#
5 + 6






# # Create a linear SVM classifier
# clf = svm.SVC(kernel='linear', C=1)
# #
# #
# # Train classifier
# clf.fit(X_train, y_train)
# #
# #
# # Make predictions on unseen test data
# clf_predictions = clf.predict(X_test)
# print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))
# #
# #
# # Check which congestion
# misclassified = np.where(y_test != clf.predict(X_test))
# #
# #
# list_of_frequency_of_missclassified_cong_patterns_in_testing = [0]*len(list_of_frequency_of_feasible_cong_patterns)
# list_of_frequency_of_cong_patterns_in_testing  = [0]*len(list_of_frequency_of_feasible_cong_patterns)
# list_of_frequency_of_cong_patterns_in_training  = [0]*len(list_of_frequency_of_feasible_cong_patterns)
# # list_of_frequency_of_missclassified_testing_cong_patterns_in_training_data = [0]*len(list_of_frequency_of_feasible_cong_patterns)
# #
# for i in misclassified[0]:
#     #current_missclassified_cong_pattern = X_test[misclassified[0][i],:].tolist()
#     class_number_for_current_missclassified_test_cong_pattern = y_test[i]
#     list_of_frequency_of_missclassified_cong_patterns_in_testing[class_number_for_current_missclassified_test_cong_pattern] += 1
# #
# #
# for i in y_test:
#     list_of_frequency_of_cong_patterns_in_testing[i] += 1
# #
# for i in y_train:
#     list_of_frequency_of_cong_patterns_in_training[i] += 1
# #
# #
# print('Frequency of All classes:                           {}'.format(list_of_frequency_of_feasible_cong_patterns))
# print('Frequency of classes in Training data:              {}'.format(list_of_frequency_of_cong_patterns_in_training))
# print('Frequency of classes in Testing data:               {}'.format(list_of_frequency_of_cong_patterns_in_testing))
# print('Frequency of missclssified classes in Testing data: {}'.format(list_of_frequency_of_missclassified_cong_patterns_in_testing))
# #
# #
# # forecasting error = sum(list_of_frequency_of_missclassified_cong_patterns_in_testing) * 100 / sum(list_of_frequency_of_cong_patterns_in_testing)
# #
# #
# 5 + 6

