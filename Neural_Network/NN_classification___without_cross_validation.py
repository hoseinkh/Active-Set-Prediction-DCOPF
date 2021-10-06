# This file trains the NN for the classification of the congestion patterns!
#
import numpy as np
import pandas as pd
#
import tensorflow
# import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
#
import os
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
# df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_branch_and_gen_patterns_of_samples___feasible.csv', sep=',',header=None)
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
input___num_of_branches = len(All_matrix_of_congestion_patterns_feasible[0])
#
#
sample_and_label_data = np.concatenate((All_feasible_samples, All_labels_feasible_matrix_of___basis_vector_labeling, All_labels_feasible_vector_of___single_number_labeling), axis = 1)
np.random.shuffle(sample_and_label_data) # Shuffles the rows
All_feasible_samples___after_shuffle                                       = sample_and_label_data[:,0:num_of_RPPs]
All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle      = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_distinctive_classes]
All_labels_feasible_vector_of___single_number_labeling___after_shuffle     = sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:].reshape(len(sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:]),)
#
# normalize the data
scaler = MinMaxScaler(feature_range = (0,1),copy = True)
scaled_All_feasible_samples = scaler.fit_transform(All_feasible_samples___after_shuffle.reshape(-1,1)).reshape( num_of_samples_with_feasible_solution  , num_of_RPPs)
#
#
#
_ , for_model___num_of_training_samples = scaled_All_feasible_samples.shape
#
#
model = Sequential([
  # Dense(My_Network.num_of_RPPs, activation='relu', input_shape=(for_model___num_of_training_samples , My_Network.num_of_RPPs)),
  Dense(max(input___num_of_branches , num_of_distinctive_classes), activation='relu', input_dim=num_of_RPPs),
  # Dense(My_Network.num_of_RPPs, activation='relu', input_shape=(1 , )),
  # Dense(My_Network.num_of_RPPs, activation='relu', input_dim=My_Network.num_of_RPPs),
  # Dense(My_Network.num_of_RPPs, activation='relu', input_dim=1),
  Dense(2*max(input___num_of_branches , num_of_distinctive_classes), activation='relu'),
  Dense(2*max(input___num_of_branches , num_of_distinctive_classes), activation='relu'),
  Dense(2*max(input___num_of_branches , num_of_distinctive_classes), activation='relu'),
  Dense(2*max(input___num_of_branches , num_of_distinctive_classes), activation='relu'),
  Dense(num_of_distinctive_classes, activation='softmax'),
])
#
model.summary()
#
# setting the parameter of the optimizer:
tensorflow.keras.optimizers.Adam(learning_rate=0.00001)
#
# define the training method
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
#
# Training the model.
model.fit(scaled_All_feasible_samples, All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle, epochs=200, validation_split = 0.20, shuffle=True, verbose = 2, batch_size = 40)
# validation_split: specifies the portion of the input data that is used for validation!
# the validation data would be the last validation_split% of the scaled_All_feasible_samples, so regardless of shaffelling this ...
# ... set remain fixed!




