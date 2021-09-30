# This file uses SVM to classify both the branches and generator patterns.
# Each pattern is classified as a hole.
#
# by generator pattern we mean whether the generator is at max capacity or not.
#
#
import numpy as np
import pandas as pd
import timeit
from ClassHmatpower import Hmatpower
import pickle
#

#
import sys, os
# import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import class_weight
from xgboost import XGBClassifier
#
# num_of_total_samples_to_consider = 55276
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
num_of_top_classes_to_consider = 41# 319   # avoid training on less populated classes (improve speed)
num_of_candidates_to_return = 1      # return multiple labels (improve accuracy)
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
## this part is specific to XGBoost: We need to make labels from 0 to #classes-1, instead of from 1 to #classes
All_labels_feasible_vector_of___single_number_labeling = np.array([i[0]-1 for i in All_labels_feasible_vector_of___single_number_labeling.tolist()]).reshape(len(All_labels_feasible_vector_of___single_number_labeling),1)
labels_of_distinctive_branch_and_gen_patterns___single_number_labeling = np.array([i-1 for i in labels_of_distinctive_branch_and_gen_patterns___single_number_labeling.tolist()[0]]).reshape(1,len(labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0]))
#
# array of number of samples that has the same congestion pattern as each distinctive congestion patterns in <matrix_of_distinctive_branch_and_gen_pattern>
# df = pd.read_csv(dir_of_saved_ndarray_files + 'array_of_num_of_patterns_for_each_dist_cong_pttrn.csv', sep=',',header=None)
# array_of_num_of_patterns_for_each_dist_cong_pttrn = df.values
array_of_num_of_patterns_for_each_dist_cong_pttrn = [0 for i in range(0,len(labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()))]
for i in All_labels_feasible_vector_of___single_number_labeling:
    print(i)
    array_of_num_of_patterns_for_each_dist_cong_pttrn[i[0]] += 1
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
#
#
# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, test_size=0.2,
                                                    random_state=0)
#
#
# min_population_accepted = 5000
array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training = [0 for i in range(0,len(labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()))]
for i in y_train:
    print(i)
    array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[int(i)] += 1
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
    if array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[curr_class] >= min_population_accepted:
        set_of_accepted_classes.add(curr_class)
list_of_accepted_classes = sorted(list(set_of_accepted_classes))
dic_mapping_accept_classes_to_new_labels  = dict()
dic_mapping_new_labels_to_original_labels = dict()
for i in range(0,len(list_of_accepted_classes)):
    dic_mapping_accept_classes_to_new_labels[list_of_accepted_classes[i]] = i
    dic_mapping_new_labels_to_original_labels[i] = list_of_accepted_classes[i]
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
if use_weighted_classififcation:
    y_train_trimmed_unique = np.unique(y_train_trimmed)
    class_weights_list = (class_weight.compute_class_weight('balanced', y_train_trimmed_unique, y_train_trimmed)).tolist()
    # class_weights = {i:0 for i in labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()}
    class_weights_dict = {y_train_trimmed_unique[i]:class_weights_list[i] for i in range(0,len(y_train_trimmed_unique))}
    for i in labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist():
        if i not in class_weights_dict.keys():
            class_weights_dict[i] = 0
    # Create a linear SVM classifier
    clf = svm.SVC(kernel='linear', C=1, class_weight=class_weights_dict)
    # Train classifier
    clf.fit(X_train_trimmed, y_train_trimmed)
else:
    # Create a XGBoost classifier model
    clf = XGBClassifier(use_label_encoder=False)
    # clf = XGBClassifier(use_label_encoder=False,
    #                     n_estimators=100,
    #                     reg_lambda=1,
    #                     gamma=0,
    #                     max_depth=3)
    # with open("XgboostCLF.pkl", 'wb') as file:
    #     pickle.dump(clf, file)
    # #
    # with open("XgboostCLF.pkl", 'rb') as file:
    #     clf = pickle.load(file)
    # Train classifier
    clf.fit(X_train_trimmed, y_train_trimmed)
#
#
# Make predictions on unseen test data
y_pred_trimmed = clf.predict(X_test_trimmed)
#
#
# confusion matrix
cm = confusion_matrix(y_test_trimmed, y_pred_trimmed)
print("Confusion Matrix : \n", cm)  # for more info on the confusion matrix see the following link:
# ... https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
#
score_on_trimmed_test_set = clf.score(X_test_trimmed, y_test_trimmed)
overall_score = score_on_trimmed_test_set*len(y_test_trimmed) / len(y_test)
print("Accuracy: {}%".format(overall_score * 100))
#
print("min_population_accepted = {}".format(min_population_accepted))
print("ratio_of_discarded_training_samples = {}%".format(100*(len(y_train)-len(y_train_trimmed))/len(y_train)))
print("ratio_of_discarded_testing_samples = {}%".format(100*(len(y_test)-len(y_test_trimmed))/len(y_test)))
print("forecasting_error = {}%".format(100 - overall_score * 100))
# print("Accuracy: {}%".format(clf.score(X_test_trimmed, y_test_trimmed) * 100))
#
qqqq = 5 + 6
#
# # # # Here we analyze for which samples the SVM does not work!
# Check which congestion
#
#
#
#
#
#
# ---------------------------------------------------------------------------------------------
## multi-candidate prediction:
#  Here we are return multiple class returns, not only one.
#  ... Hence, we return the, e.g., top three classes with highest scores
print("-------------------------------------------------------")
start = timeit.default_timer()
overall_scores = clf.predict_proba(X_test_trimmed)
max_score___multi_candidate = []
new_Hmatpower_object = Hmatpower(case_number = "ClassHmatpower_for_IEEE_162_bus", gen_DA_RT_list = ["D" for i in range(0,12)], epsH=0.0001, probability_threshold_r1=0.9, probability_threshold_r2=0.9, probability_threshold_r3=0.9, ratio_std_to_mean_r1=0.2, ratio_std_to_mean_r2=0.2, ratio_std_to_mean_r3=0.2)
PTDF_inverse = np.random.rand(284,162)
for i in range(0,X_test_trimmed.shape[0]):
    if len(set_of_accepted_classes) == 2:
        scores_for_curr_sample = overall_scores[i]
        if scores_for_curr_sample > 0:
            max_score___multi_candidate.append(2)
        else:
            max_score___multi_candidate.append(1)
    else:
        scores_for_curr_sample = overall_scores[i,:].tolist()
        zzzz = sorted(range(len(scores_for_curr_sample)), key=lambda i: scores_for_curr_sample[i])
        zzzz = [i for i in zzzz]
        zzzz.reverse()
        max_score___multi_candidate.append(zzzz[0:num_of_candidates_to_return])
max_score___multi_candidate = np.array(max_score___multi_candidate)
# np.concatenate((y_pred.reshape(len(y_pred),1), max_score___multi_candidate), axis=1)
#
if use_multi_candidate_scroing_approach:
    count_correct_classification_multi_class = 0
    for i in range(0,len(y_test_trimmed)):
        # print("i = " + str(i))
        print("===> " + str(i * 100 / len(y_test_trimmed)))
        true_class_trimmed = y_test_trimmed[i]
        if len(set_of_accepted_classes) == 2:
            if true_class_trimmed == max_score___multi_candidate[i]:
                count_correct_classification_multi_class += 1
        else:
            if true_class_trimmed in max_score___multi_candidate[i,:].tolist():
                count_correct_classification_multi_class += 1
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

5 + 6
