# This file uses LDA to classify congestions. Each congestion pattern is classified as a hole.
#
import numpy as np
import pandas as pd
import timeit
#
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.covariance import OAS
#
from ClassHmatpower import Hmatpower
#
import os
#
#
num_of_top_classes_to_consider = 319
num_of_candidates_to_return = 15
use_multi_candidate_scroing_approach = True
#
# if this variable is set to <True>, the code uses cross validation. If this variable ...
# ... is set to <False>, then it applies the LDA algorithm once, and there is no cross validation
#
#
## Step 1: Reading the data of scenarios! These data are stored in the directory named <Data>, and are of .csv format!
#
curr_dir = os.getcwd()
dir_of_saved_ndarray_files = curr_dir + '/Data/'
# dir_of_saved_ndarray_files = curr_dir + '/'
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
#
# # Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, test_size=0.2, random_state=0)
#
#
# for all data
array_of_num_of_patterns_for_each_dist_cong_pttrn_in_ALL_data = [0 for i in range(0, len(labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()))]
for i in All_labels_feasible_vector_of___single_number_labeling___after_shuffle:
    array_of_num_of_patterns_for_each_dist_cong_pttrn_in_ALL_data[int(i) - 1] += 1
#
# for training
array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training = [0 for i in range(0, len(labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()))]
for i in y_train:
    array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[int(i) - 1] += 1
#
# for testing
array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing = [0 for i in range(0, len(labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()))]
for i in y_test:
    array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing[int(i) - 1] += 1
#
ppp = sorted(list(set(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training)))
ppp.reverse()
try:
    min_population_accepted = ppp[num_of_top_classes_to_consider - 1]
except IndexError:
    min_population_accepted = ppp[-1]
set_of_accepted_classes = set()
for i in range(0, X_train.shape[0]):
    curr_class = int(y_train[i])
    if array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[curr_class - 1] >= min_population_accepted:
        set_of_accepted_classes.add(curr_class)
list_of_accepted_classes = sorted(list(set_of_accepted_classes))
dic_mapping_accept_classes_to_new_labels = dict()
dic_mapping_new_labels_to_original_labels = dict()
for i in range(0, len(list_of_accepted_classes)):
    dic_mapping_accept_classes_to_new_labels[list_of_accepted_classes[i]] = i + 1
    dic_mapping_new_labels_to_original_labels[i + 1] = list_of_accepted_classes[i]
#
y_train_trimmed = []
X_train_trimmed = []
for i in range(0, len(y_train)):
    curr_label = int(y_train[i])
    if curr_label in set_of_accepted_classes:
        y_train_trimmed.append(dic_mapping_accept_classes_to_new_labels[curr_label])
        X_train_trimmed.append(X_train[i, :].tolist())
y_train_trimmed = np.array(y_train_trimmed)
X_train_trimmed = np.array(X_train_trimmed)
#
#
y_test_trimmed = []
X_test_trimmed = []
num_of_samples_with_discarded_class_labels = 0
for i in range(0, len(y_test)):
    curr_label = int(y_test[i])
    if curr_label in set_of_accepted_classes:
        y_test_trimmed.append(dic_mapping_accept_classes_to_new_labels[curr_label])
        X_test_trimmed.append(X_test[i, :].tolist())
    else:
        num_of_samples_with_discarded_class_labels += 1
y_test_trimmed = np.array(y_test_trimmed)
X_test_trimmed = np.array(X_test_trimmed)
#
#
#
# # Create LDA
# lda = LDA(solver='eigen', tol = 0.000001)
# lda = LDA(solver='lsqr', tol = 0.000001)
# lda = LDA(solver='lsqr', tol = 0.000001, shrinkage='auto')
## very good result
# oa = OAS(store_precision=False, assume_centered=False)
# lda = LDA(solver='lsqr', tol = 0.000001, covariance_estimator=oa)
##
# lda = LDA()
oa = OAS(store_precision=False, assume_centered=False)
lda = LDA(solver='lsqr', tol = 0.00000001, covariance_estimator=oa)

# lda = LDA(solver='eigen', tol = 0.000001, shrinkage='auto')
# lda = LDA(solver='lsqr', tol = 0.000001, shrinkage='auto')
#
# # Train classifier
lda.fit(X_train_trimmed, y_train_trimmed)
#
# Make predictions on unseen test data
y_pred_trimmed = lda.predict(X_test_trimmed)
#
## Note that the following is not the exact score for the LDA
#print(str(lda.score(X_test_trimmed, y_test_trimmed) * 100) + '%')
#
# the following is the exact score for the LDA
print("{}%".format(100*sum([y_test_trimmed[i] == y_pred_trimmed[i] for i in range(0, len(y_test_trimmed))]) / len(y_test)))
#
print("-------------------------------------------------------")
start = timeit.default_timer()
overall_scores = lda.predict_proba(X_test_trimmed)
max_score___multi_candidate = []
new_Hmatpower_object = Hmatpower(case_number = "ClassHmatpower_for_IEEE_162_bus", gen_DA_RT_list = ["D" for i in range(0,12)], epsH=0.0001, probability_threshold_r1=0.9, probability_threshold_r2=0.9, probability_threshold_r3=0.9, ratio_std_to_mean_r1=0.2, ratio_std_to_mean_r2=0.2, ratio_std_to_mean_r3=0.2)
PTDF_inverse = np.random.rand(284,162)
list_of_num_of_samples_not_corectly_classified___for_each_class_in_testing = [0 for i in range(0,All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle.shape[1])]
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
        zzzz = [(i+1) for i in zzzz]
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
            else:  # we have missclassification
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
#
list_of_percentage_samples_not_corectly_classified___for_each_class_in_testing = []
for i in range(0,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing)):
    if array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing[i] != 0:
        list_of_percentage_samples_not_corectly_classified___for_each_class_in_testing.append(100*list_of_num_of_samples_not_corectly_classified___for_each_class_in_testing[i] / array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing[i])
    else:
        list_of_percentage_samples_not_corectly_classified___for_each_class_in_testing.append(0)
#
list_of_probabilities_of_each_class_in_testing = [100*array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing[i] / sum(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing) for i in range(0,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing))]
list_of_probabilities_of_each_class_in_training = [100*array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[i] / sum(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training) for i in range(0,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training))]
#
tuple_of_num_of_patterns_for_each_dist_cong_pttrn_and_class_index_in_testing  = [(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing[i],  int(i)+1) for i in range(len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing))]
tuple_of_num_of_patterns_for_each_dist_cong_pttrn_and_class_index_in_training = [(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[i], int(i)+1) for i in range(len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training))]
#
class_index_from_least_to_most_likely_in_testing  = [i[1] for i in sorted(tuple_of_num_of_patterns_for_each_dist_cong_pttrn_and_class_index_in_testing, key=lambda i: i[0])]
class_index_from_least_to_most_likely_in_training = [i[1] for i in sorted(tuple_of_num_of_patterns_for_each_dist_cong_pttrn_and_class_index_in_training, key=lambda i: i[0])]
#

list_of_percentage_samples_not_corectly_classified_in_testing_data___for_each_class_sorted_from_least_likely_to_most_likely_in_training_data = [list_of_percentage_samples_not_corectly_classified___for_each_class_in_testing[i-1] for i in class_index_from_least_to_most_likely_in_training]

5 + 6



## naive prediction, top hhh cases
# hhh = 5; sum([array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing[n] for n in [m[1] for m in [sorted([(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[i], i) for i in range(len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training))], key = lambda j: j[0], reverse = True)[k] for k in range(hhh)]]]) / sum(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_testing)







