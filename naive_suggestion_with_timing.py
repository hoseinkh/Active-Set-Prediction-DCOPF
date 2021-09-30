# This file uses SVM to classify both the branches and generator patterns.
# Each pattern is classified as a hole.
#
# by generator pattern we mean whether the generator is at max capacity or not.
#
#
import numpy as np
import pandas as pd
import timeit
# import sklearn
import sklearn.model_selection
#
from ClassHmatpower import Hmatpower
#
# import sklearn.external.joblib as extjoblib
# import joblib
#
import sys, os
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split # , GridSearchCV
# from sklearn.utils import class_weight
#
num_of_total_samples_to_consider = 55276
#
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import cross_val_score
# import statistics
# from sklearn.metrics import confusion_matrix
#
#
use_cross_validation = False
use_weighted_classififcation = False
use_multi_candidate_scroing_approach = True
num_of_top_classes_to_consider = 29   # avoid training on less populated classes (improve speed)
num_of_candidates_to_return = 29      # return multiple labels (improve accuracy)
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
#
# array of number of samples that has the same congestion pattern as each distinctive congestion patterns in <matrix_of_distinctive_branch_and_gen_pattern>
# df = pd.read_csv(dir_of_saved_ndarray_files + 'array_of_num_of_patterns_for_each_dist_cong_pttrn.csv', sep=',',header=None)
# array_of_num_of_patterns_for_each_dist_cong_pttrn = df.values
array_of_num_of_patterns_for_each_dist_cong_pttrn = [0 for i in range(0,len(labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()))]
for i in All_labels_feasible_vector_of___single_number_labeling:
    print(i)
    array_of_num_of_patterns_for_each_dist_cong_pttrn[i[0]-1] += 1
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
# num_of_branches = 284
# num_of_gen      = 12
#
#
sample_and_label_data = np.concatenate((All_feasible_samples, All_labels_feasible_matrix_of___basis_vector_labeling, All_labels_feasible_vector_of___single_number_labeling), axis = 1)
np.random.seed(0)
np.random.shuffle(sample_and_label_data) # Shuffles the rows
All_feasible_samples___after_shuffle                                       = sample_and_label_data[:,0:num_of_RPPs]
All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle      = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_distinctive_classes]
All_labels_feasible_vector_of___single_number_labeling___after_shuffle     = sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:].reshape(len(sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:]),)
#
#
#
# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, test_size=0.2,
                                                    random_state=0)
#
#
# min_population_accepted = 5000
# for training data
array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training = [0 for i in range(0,len(labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()))]
for i in y_train:
    # print(i)
    array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[int(i)-1] += 1
# for testing data
array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing = [0 for i in range(0,len(labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()))]
for i in y_test:
    # print(i)
    array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing[int(i)-1] += 1
#
#
array_of_num_of_patterns_with_class_number_for_each_dist_cong_pttrn_in_training = [(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[int(i)-1], i) for i in range(1,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training)+1)]
sorted_by_population_array_of_num_of_patterns_with_class_number_for_each_dist_cong_pttrn_in_training = sorted(array_of_num_of_patterns_with_class_number_for_each_dist_cong_pttrn_in_training, key=lambda tup: tup[0])
sorted_by_population_array_of_num_of_patterns_with_class_number_for_each_dist_cong_pttrn_in_training.reverse()
#
ppp = sorted(list(set(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training)))
ppp.reverse()
try:
    min_population_accepted = ppp[num_of_top_classes_to_consider - 1]
except IndexError:
    min_population_accepted = ppp[-1]
set_of_accepted_classes = set()
for i in range(0,X_train.shape[0]):
    curr_class = int(y_train[i])
    if array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[curr_class-1] >= min_population_accepted:
        set_of_accepted_classes.add(curr_class)
list_of_accepted_classes = sorted(list(set_of_accepted_classes))
dic_mapping_accept_classes_to_new_labels  = dict()
dic_mapping_new_labels_to_original_labels = dict()
for i in range(0,len(list_of_accepted_classes)):
    dic_mapping_accept_classes_to_new_labels[list_of_accepted_classes[i]] = i+1
    dic_mapping_new_labels_to_original_labels[i+1] = list_of_accepted_classes[i]
#
y_train_trimmed = []
X_train_trimmed = []
for i in range(0,len(y_train)):
    curr_label = int(y_train[i])
    if curr_label in set_of_accepted_classes:
        y_train_trimmed.append(dic_mapping_accept_classes_to_new_labels[curr_label])
        X_train_trimmed.append(X_train[i,:].tolist())
y_train_trimmed = np.array(y_train_trimmed)
X_train_trimmed = np.array(X_train_trimmed)
#
#
y_test_trimmed = []
X_test_trimmed = []
num_of_samples_with_discarded_class_labels = 0
for i in range(0,len(y_test)):
    curr_label = int(y_test[i])
    if curr_label in set_of_accepted_classes:
        y_test_trimmed.append(dic_mapping_accept_classes_to_new_labels[curr_label])
        X_test_trimmed.append(X_test[i,:].tolist())
    else:
        num_of_samples_with_discarded_class_labels += 1
y_test_trimmed = np.array(y_test_trimmed)
X_test_trimmed = np.array(X_test_trimmed)
#
#
# y_test_trimmed = y_test
# X_test_trimmed = X_test
#
# Make predictions on unseen test data
y_pred_trimmed = [sorted_by_population_array_of_num_of_patterns_with_class_number_for_each_dist_cong_pttrn_in_training[0][1] for i in range(0,X_test_trimmed.shape[0])] #
#
#
#
qqqq = 5 + 6
#
#
# ---------------------------------------------------------------------------------------------
## multi-candidate prediction:
#  Here we are return multiple class returns, not only one.
#  ... Hence, we return the, e.g., top three classes with highest scores
print("-------------------------------------------------------")
start = timeit.default_timer()
# overall_scores = clf.decision_function(X_test_trimmed)
overall_scores = np.repeat(np.array([i[1] for i in sorted_by_population_array_of_num_of_patterns_with_class_number_for_each_dist_cong_pttrn_in_training][0:num_of_candidates_to_return]).reshape(1,num_of_candidates_to_return), X_test_trimmed.shape[0], axis = 0)
top_classes___multi_candidate = []
new_Hmatpower_object = Hmatpower(case_number = "ClassHmatpower_for_IEEE_162_bus", gen_DA_RT_list = ["D" for i in range(0,12)], epsH=0.0001, probability_threshold_r1=0.9, probability_threshold_r2=0.9, probability_threshold_r3=0.9, ratio_std_to_mean_r1=0.2, ratio_std_to_mean_r2=0.2, ratio_std_to_mean_r3=0.2)
PTDF_inverse = np.random.rand(284,162)
for i in range(0,X_test_trimmed.shape[0]):
    scores_for_curr_sample = overall_scores[i, :].tolist()
    top_classes___multi_candidate.append(scores_for_curr_sample[0:num_of_candidates_to_return])
top_classes___multi_candidate = np.array(top_classes___multi_candidate)
# np.concatenate((y_pred.reshape(len(y_pred),1), max_score___multi_candidate), axis=1)
#
list_of_num_of_samples_not_corectly_classified___for_each_class_in_testing = [0 for i in range(0,All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle.shape[1])]
if use_multi_candidate_scroing_approach:
    count_correct_classification_multi_class = 0
    for i in range(0,len(y_test_trimmed)):
        # print("i = " + str(i))
        print("===> " + str(i * 100 / len(y_test_trimmed)))
        true_class_trimmed = y_test_trimmed[i]
        if true_class_trimmed in top_classes___multi_candidate[i,:].tolist():
            count_correct_classification_multi_class += 1
        else: # we have missclassification
            true_class_trimmed___original_label = dic_mapping_new_labels_to_original_labels[true_class_trimmed]
            list_of_num_of_samples_not_corectly_classified___for_each_class_in_testing[int(true_class_trimmed___original_label) - 1] += 1
        # for estimating time
        for j in range(0, num_of_candidates_to_return):  # for estimating the time
            ## for estimating the running time
            # estimating the time for calculating the output of generators
            np.matmul(PTDF_inverse[0, :], X_test_trimmed[i, :]) < new_Hmatpower_object.branch[:, 7]
            # estimating the time for verifying the feasibility
            np.matmul(PTDF_inverse, X_test_trimmed[i, :]) < new_Hmatpower_object.branch[:, 7]
            np.matmul(PTDF_inverse, X_test_trimmed[i, :]) < new_Hmatpower_object.branch[:, 7]
            # new_Hmatpower_object.bus[:,2] = X_test_trimmed[i,:]
            # new_Hmatpower_object.check_DA_feasibility()
            _ = min(np.random.rand(1, num_of_candidates_to_return).tolist())  # to get estimate of time
    # accuracy_multi_candidate = count_correct_classification_multi_class/len(y_test)
    # print("accuracy_multi_candidate = {}%".format(accuracy_multi_candidate * 100))
    overall_accuracy_multi_candidate = count_correct_classification_multi_class / len(y_test)
    print("overall_accuracy_multi_candidate = {}%".format(overall_accuracy_multi_candidate*100))
#
stop = timeit.default_timer()
print('Computational time = {}'.format(stop - start))
#
list_of_percentage_samples_not_corectly_classified___for_each_class_in_testing = []
for i in range(0,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing)):
    if array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing[i] != 0:
        list_of_percentage_samples_not_corectly_classified___for_each_class_in_testing.append(100*list_of_num_of_samples_not_corectly_classified___for_each_class_in_testing[i] / array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing[i])
    else:
        list_of_percentage_samples_not_corectly_classified___for_each_class_in_testing.append(0)
#
list_of_probabilities_of_each_class_in_testing = [100*array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing[i] / sum(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing) for i in range(0,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing))]
#
5 + 6
