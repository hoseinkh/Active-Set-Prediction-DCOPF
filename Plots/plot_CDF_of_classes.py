#
#
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#
curr_dir = os.getcwd()
dir_of_saved_ndarray_files = curr_dir + '/Data/'

# matrix of all feasible samples!
df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_feasible_samples.csv', sep=',',header=None)
All_feasible_samples = df.values

# vector of single number labels!
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_feasible_samples___single_number_labeling.csv', sep=',',header=None)
All_labels_feasible_vector_of___single_number_labeling = df.values
#
# matrix of basis vector labels!
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_feasible_samples___basis_vector_labeling.csv', sep=',',header=None)
All_labels_feasible_matrix_of___basis_vector_labeling = df.values
#
num_of_samples_with_feasible_solution , num_of_RPPs = All_feasible_samples.shape
_ , num_of_distinctive_classes = All_labels_feasible_matrix_of___basis_vector_labeling.shape
#
# single-number labels corresponding to the distinctive congestion patterns in <matrix_of_distinctive_branch_and_gen_pattern>
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_dist_feasible_samples___single_number_labeling.csv', sep=',',header=None)
labels_of_distinctive_branch_and_gen_patterns___single_number_labeling = df.values
#
# # basis-vector labels corresponding to the distinctive congestion patterns in <matrix_of_distinctive_branch_and_gen_pattern>
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_dist_feasible_samples___basis_vector_labeling.csv', sep=',',header=None)
labels_of_distinctive_branch_and_gen_patterns___basis_vector_labeling = df.values
#
sample_and_label_data = np.concatenate((All_feasible_samples, All_labels_feasible_matrix_of___basis_vector_labeling, All_labels_feasible_vector_of___single_number_labeling), axis = 1)
np.random.seed(0)
np.random.shuffle(sample_and_label_data)
All_feasible_samples___after_shuffle                                       = sample_and_label_data[:,0:num_of_RPPs]
All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle      = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_distinctive_classes]
All_labels_feasible_vector_of___single_number_labeling___after_shuffle     = sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:].reshape(len(sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:]),)
#
# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(All_feasible_samples___after_shuffle, All_labels_feasible_vector_of___single_number_labeling___after_shuffle, test_size=0.2,
                                                    random_state=0)
#
# array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training = [1046, 8421, 8212, 2315, 75, 6281, 4500, 860, 41, 1445, 329, 514, 65, 11, 1462, 15, 1435, 997, 354, 74, 41, 1, 9, 130, 76, 13, 342, 64, 71, 686, 10, 82, 24, 20, 2, 207, 34, 134, 133, 31, 256, 147, 35, 139, 7, 58, 114, 187, 6, 48, 7, 316, 216, 10, 7, 108, 91, 2, 16, 0, 16, 5, 32, 117, 53, 1, 2, 73, 13, 2, 139, 4, 29, 24, 6, 69, 1, 43, 20, 57, 2, 14, 5, 1, 23, 16, 18, 25, 8, 61, 0, 2, 27, 16, 3, 29, 1, 35, 17, 1, 1, 29, 46, 7, 16, 27, 1, 4, 14, 45, 7, 19, 8, 7, 5, 2, 4, 8, 5, 12, 2, 7, 1, 2, 2, 31, 4, 2, 10, 11, 28, 4, 10, 1, 1, 3, 9, 2, 2, 7, 3, 41, 7, 16, 19, 6, 7, 1, 1, 0, 1, 5, 3, 2, 9, 2, 2, 9, 1, 6, 3, 2, 6, 3, 15, 4, 1, 5, 2, 2, 1, 2, 1, 11, 2, 2, 2, 5, 18, 6, 5, 3, 12, 3, 2, 2, 2, 3, 5, 0, 12, 1, 2, 2, 2, 1, 2, 1, 5, 4, 1, 1, 1, 1, 1, 4, 2, 3, 2, 1, 2, 5, 7, 1, 2, 0, 1, 1, 2, 1, 4, 0, 2, 1, 2, 1, 3, 1, 2, 1, 3, 1, 1, 5, 1, 2, 0, 3, 1, 2, 2, 3, 1, 1, 2, 1, 1, 0, 1, 1, 1, 3, 3, 2, 0, 0, 3, 1, 2, 2, 1, 4, 1, 1, 0, 1, 1, 0, 1, 1, 2, 2, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 2, 1, 3, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 1, 0, 1, 1, 2, 1, 0, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1]
array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training = [0 for i in range(0,len(labels_of_distinctive_branch_and_gen_patterns___single_number_labeling[0].tolist()))]
for i in y_train:
    print(i)
    array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[int(i)-1] += 1
array_num_patterns_with_index = [(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training[i], i) for i in range(0,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training))]
sorted_array_num_patterns_with_index = sorted(array_num_patterns_with_index, key= lambda i: i[0])
sorted_array_num_patterns_with_index.reverse()
cumulative_probability = [sorted_array_num_patterns_with_index[0]]
for i in range(1,len(sorted_array_num_patterns_with_index)):
    cumulative_probability.append(( (cumulative_probability[i-1][0] + sorted_array_num_patterns_with_index[i][0]), sorted_array_num_patterns_with_index[i][1]))
total_num_samples = sum(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training)
cumulative_probability = [(cumulative_probability[i][0]/total_num_samples, cumulative_probability[i][1]) for i in range(0,len(cumulative_probability))]
#
# plt.bar(list(range(0,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training))), [i[0] for i in cumulative_probability])
# plt.subplots(figsize=(4,2))
HHH = [i for i in plt.rcParams["figure.figsize"]]
plt.rcParams["figure.figsize"] = [HHH[0], 0.5*HHH[1]]
plt.plot(list(range(0,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training))), [i[0] for i in cumulative_probability], linestyle='None', marker='*')
plt.xlabel('Active sets sorted from most likely to least likely', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
# plt.plot(list(range(0,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training))), [i[0]/total_num_samples for i in sorted_array_num_patterns_with_index])
#
# plt.bar(list(range(0,len(array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training))), [i/total_num_samples for i in array_of_num_of_patterns_for_each_dist_cong_pttrn_in_training])
# plt.gca().set_yscale('log')
# plt.xlabel('probability')
# plt.ylabel('value')


5 + 6







