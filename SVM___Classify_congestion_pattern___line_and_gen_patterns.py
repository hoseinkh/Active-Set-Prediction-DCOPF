# This file uses SVM to classify both the branches and generator patterns.
# Each pattern is classified as a hole.
#
# by generator pattern we mean whether the generator is at max capacity or not.
#
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
from sklearn.utils import class_weight
#
#
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import statistics
from sklearn.metrics import confusion_matrix
#
#
use_cross_validation = False
use_weighted_classififcation = False
use_multi_candidate_scroing_approach = True
num_of_candidates_to_return = 2
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
# matrix of actual branch and gen patterns corresponding to the feasible samples!
df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_branch_and_gen_patterns_of_samples___feasible.csv', sep=',',header=None)
All_matrix_of_branch_and_gen_patterns_feasible = df.values
#
# vector of single number labels!
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_feasible_samples___single_number_labeling.csv', sep=',',header=None)
All_labels_feasible_vector_of___single_number_labeling = df.values
#
# matrix of basis vector labels!
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_feasible_samples___basis_vector_labeling.csv', sep=',',header=None)
All_labels_feasible_matrix_of___basis_vector_labeling = df.values
#
# matrix of distinctive congestion patterns!
df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_distinctive_branch_and_gen_pattern.csv', sep=',',header=None)
matrix_of_distinctive_branch_and_gen_pattern = df.values
#
# single-number labels corresponding to the distinctive congestion patterns in <matrix_of_distinctive_branch_and_gen_pattern>
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_dist_feasible_samples___single_number_labeling.csv', sep=',',header=None)
labels_of_distinctive_branch_and_gen_patterns___single_number_labeling = df.values
#
# # basis-vector labels corresponding to the distinctive congestion patterns in <matrix_of_distinctive_branch_and_gen_pattern>
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_dist_feasible_samples___basis_vector_labeling.csv', sep=',',header=None)
labels_of_distinctive_branch_and_gen_patterns___basis_vector_labeling = df.values
#
# array of number of samples that has the same congestion pattern as each distinctive congestion patterns in <matrix_of_distinctive_branch_and_gen_pattern>
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
#
num_of_branches = 284
num_of_gen      = 12
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
    # Create a linear SVM classifier
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, cv=10)
    #Frequency of All classes
    print('list of scores = {}'.format(scores))
    avg_score = sum(scores) / len(scores)
    std_score = statistics.stdev(scores)
    #
    print("Accuracy: {}% (+/- {}%)".format(avg_score*100,std_score*100))
    #
    #
else:
    #
    #
    # Split data to train and test on 80-20 ratio
    X_train, X_test, y_train, y_test = train_test_split(All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, test_size=0.2,
                                                        random_state=0)
    #
    #
    #
    #
    if use_weighted_classififcation:
        y_train_unique = np.unique(y_train)
        class_weights_list = (class_weight.compute_class_weight('balanced', y_train_unique, y_train)).tolist()
        # class_weights = {i:0 for i in labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()}
        class_weights_dict = {y_train_unique[i]:class_weights_list[i] for i in range(0,len(y_train_unique))}
        for i in labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist():
            if i not in class_weights_dict.keys():
                class_weights_dict[i] = 0
            # class_weights[i] = class_weights_list[int(i)-1]
        # for i in
        # class_weights = {i:class_weights[i-1] for i in range(1,len(class_weights)+1)}
        # Create a linear SVM classifier
        clf = svm.SVC(kernel='linear', C=1, class_weight=class_weights_dict)
        # Train classifier
        clf.fit(X_train, y_train)
    else:
        # Create a linear SVM classifier
        clf = svm.SVC(kernel='linear', C=1, probability=True, break_ties=True)
        # Train classifier
        clf.fit(X_train, y_train)
    #
    #
    # Make predictions on unseen test data
    y_pred = clf.predict(X_test)
    #
    #
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix : \n", cm)  # for more info on the confusion matrix see the following link:
    # ... https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
    #
    print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100))
    #
    #
    # # # # Here we analyze for which samples the SVM does not work!
    # Check which congestion
    #
    misclassified = np.where(
        np.subtract(y_test.reshape((len(y_test)), ), clf.predict(X_test).reshape((len(y_test)), )).reshape((len(y_test)), ) != 0)
    #
    #
    list_of_frequency_of_missclassified_branch_and_gen_patterns_in_testing = [0] * num_of_distinctive_classes
    list_of_frequency_of_branch_and_gen_patterns_in_testing = [0] * num_of_distinctive_classes
    list_of_frequency_of_branch_and_gen_patterns_in_training = [0] * num_of_distinctive_classes
    #
    #
    for i in misclassified[0]:
        class_number_for_current_missclassified_test_branch_and_gen_pattern = int(y_test[i])  # note that we know that elements of <y_test> are always integer numbers! (because they are class numbers!)
        # print("class_number_for_current_missclassified_test_branch_and_gen_pattern = {}".format(class_number_for_current_missclassified_test_branch_and_gen_pattern))
        list_of_frequency_of_missclassified_branch_and_gen_patterns_in_testing[
            class_number_for_current_missclassified_test_branch_and_gen_pattern-1] += 1
    #
    for i in y_test:
        list_of_frequency_of_branch_and_gen_patterns_in_testing[int(i) - 1] += 1
    #
    for i in y_train:
        list_of_frequency_of_branch_and_gen_patterns_in_training[int(i) - 1] += 1
    #
    #
    print('Frequency of All classes:                           {}'.format([zz[0] for zz in array_of_num_of_patterns_for_each_dist_cong_pttrn.tolist()]))
    print('Frequency of classes in Training data:              {}'.format(list_of_frequency_of_branch_and_gen_patterns_in_training))
    print('Frequency of classes in Testing data:               {}'.format(list_of_frequency_of_branch_and_gen_patterns_in_testing))
    print('Frequency of missclssified classes in Testing data: {}'.format(list_of_frequency_of_missclassified_branch_and_gen_patterns_in_testing))
    #
    #
    forecasting_error = sum(list_of_frequency_of_missclassified_branch_and_gen_patterns_in_testing) * 100 / sum(list_of_frequency_of_branch_and_gen_patterns_in_testing)
    print("forecasting_error = {}%".format(forecasting_error))
    #
    5 + 6
    #
    #
# scoring based on the probabilities
ind_based_on_probabilities = (np.argwhere(clf._predict_proba(X_test)==np.amax(clf._predict_proba(X_test),1, keepdims=True)))[:,1] + 1
predicted_y___vs___max_prob = np.concatenate((y_pred.reshape(len(y_pred),1), ind_based_on_probabilities.reshape(len(ind_based_on_probabilities),1)), axis=1)
num_of_same_predictions_for_probability_based_ranking = sum([predicted_y___vs___max_prob[i,0] == predicted_y___vs___max_prob[i,1] for i in range(0,predicted_y___vs___max_prob.shape[0])])
ratio_of_correct_forecasting___predicted_y___vs___max_prob = num_of_same_predictions_for_probability_based_ranking / len(y_pred)
#
# scoring based on the scores of each class and the decision function
# the SVC uses this method. Although there is a slight difference between the results.
# ... meaning that less than 0.5% mismatch exists for the case that I verified
# ... and the reason is that SVC looks for the first class that has the closest score
# ... to the maximum.
# If we set "break_ties=True" then the results will be perfectly matched! (see: https://scikit-learn.org/stable/modules/svm.html#scores-probabilities)
ind_based_on_scores = (np.argwhere(clf.decision_function(X_test)==np.amax(clf.decision_function(X_test),1, keepdims=True)))[:,1] + 1
predicted_y___vs___max_score = np.concatenate((y_pred.reshape(len(y_pred),1), ind_based_on_scores.reshape(len(ind_based_on_scores),1)), axis=1)
num_of_same_predictions_for_score_based_ranking = sum([predicted_y___vs___max_score[i,0] == predicted_y___vs___max_score[i,1] for i in range(0,predicted_y___vs___max_score.shape[0])])
ratio_of_correct_forecasting___predicted_y___vs___max_score = num_of_same_predictions_for_score_based_ranking / len(y_pred)
index_and_values_of_mismatches_between_predictions_and_maxscore_approach = [(i, [predicted_y___vs___max_score[i,0], predicted_y___vs___max_score[i,1]]) for i in range(0,predicted_y___vs___max_score.shape[0]) if predicted_y___vs___max_score[i,0] != predicted_y___vs___max_score[i,1]]
#
# ---------------------------------------------------------------------------------------------
## multi-candidate prediction:
#  Here we are return multiple class returns, not only one.
#  ... Hence, we return the, e.g., top three classes with highest scores
overall_scores = clf.decision_function(X_test)
max_score___multi_candidate = []
for i in range(0,X_test.shape[0]):
    scores_for_curr_sample = overall_scores[i,:].tolist()
    zzzz = sorted(range(len(scores_for_curr_sample)), key=lambda i: scores_for_curr_sample[i])
    zzzz = [i+1 for i in zzzz]
    zzzz.reverse()
    max_score___multi_candidate.append(zzzz[0:num_of_candidates_to_return])
max_score___multi_candidate = np.array(max_score___multi_candidate)
# np.concatenate((y_pred.reshape(len(y_pred),1), max_score___multi_candidate), axis=1)
#
if use_multi_candidate_scroing_approach:
    count_correct_classification_multi_class = 0
    for i in range(0,len(y_test)):
        print("i = " + str(i))
        true_class = y_test[i]
        if true_class in max_score___multi_candidate[i,:].tolist():
            count_correct_classification_multi_class += 1
    accuracy_multi_candidate = count_correct_classification_multi_class/len(y_test)
    print("accuracy_multi_candidate = {}%".format(accuracy_multi_candidate*100))
5 + 6
