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
num_of_top_classes_to_consider = 319
num_of_candidates_to_return = 15
use_multi_candidate_scroing_approach = True
#
use_cross_validation = False
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
if use_cross_validation:  # use cross validation: apply LDA multiple times
    # # Make predictions on unseen test data  +  Cross-validation error checking
    scores = cross_val_score(LDA(), All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, cv=10)
    print("Accuracy: {}% (+/- {}%)".format(scores.mean() * 100, (scores.std() * 100) * 2))
else:
    # # Split data to train and test on 80-20 ratio
    X_train, X_test, y_train, y_test = train_test_split(All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, test_size=0.2, random_state=0)
    #
    #
    #
    # min_population_accepted = 5000
    array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training = [0 for i in range(0, len(
        labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()))]
    for i in y_train:
        print(i)
        array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[int(i) - 1] += 1
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
    lda = LDA(solver='eigen', tol = 0.000001)
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
    overall_scores = lda.decision_function(X_test_trimmed)
    max_score___multi_candidate = []
    for i in range(0, X_test_trimmed.shape[0]):
        if len(set_of_accepted_classes) == 2:
            scores_for_curr_sample = overall_scores[i]
            if scores_for_curr_sample > 0:
                max_score___multi_candidate.append(2)
            else:
                max_score___multi_candidate.append(1)
        else:
            scores_for_curr_sample = overall_scores[i, :].tolist()
            zzzz = sorted(range(len(scores_for_curr_sample)), key=lambda i: scores_for_curr_sample[i])
            zzzz = [i + 1 for i in zzzz]
            zzzz.reverse()
            max_score___multi_candidate.append(zzzz[0:num_of_candidates_to_return])
    max_score___multi_candidate = np.array(max_score___multi_candidate)
    # np.concatenate((y_pred.reshape(len(y_pred),1), max_score___multi_candidate), axis=1)
    #
    if use_multi_candidate_scroing_approach:
        count_correct_classification_multi_class = 0
        for i in range(0, len(y_test_trimmed)):
            # print("i = " + str(i))
            true_class_trimmed = y_test_trimmed[i]
            if len(set_of_accepted_classes) == 2:
                if true_class_trimmed == max_score___multi_candidate[i]:
                    count_correct_classification_multi_class += 1
            else:
                if true_class_trimmed in max_score___multi_candidate[i, :].tolist():
                    count_correct_classification_multi_class += 1
        # accuracy_multi_candidate = count_correct_classification_multi_class/len(y_test)
        # print("accuracy_multi_candidate = {}%".format(accuracy_multi_candidate * 100))
        overall_accuracy_multi_candidate = count_correct_classification_multi_class / len(y_test)
        print("overall_accuracy_multi_candidate = {}%".format(overall_accuracy_multi_candidate * 100))
    temp_var = 5 + 6
    #
    # # # # Here we analyze for which samples the LDA does not work!
    # Check which congestion
    #
    if False:
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
        print('Frequency of All classes:                           {}'.format(array_of_num_of_patterns_for_each_dist_cong_pttrn))
        print('Frequency of classes in Training data:              {}'.format(list_of_frequency_of_cong_patterns_in_training))
        print('Frequency of classes in Testing data:               {}'.format(list_of_frequency_of_cong_patterns_in_testing))
        print('Frequency of missclssified classes in Testing data: {}'.format(list_of_frequency_of_missclassified_cong_patterns_in_testing))
        #

        forecasting_error = sum(list_of_frequency_of_missclassified_cong_patterns_in_testing) * 100 / sum(list_of_frequency_of_cong_patterns_in_testing)
        print("forecasting_error = {}%".format(forecasting_error))


