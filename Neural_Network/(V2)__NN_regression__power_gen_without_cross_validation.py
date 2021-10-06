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
from sklearn.model_selection import train_test_split
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
labels_of_distinctive_congestion_patterns___single_number_labeling = df.values[0]
#
# # basis-vector labels corresponding to the distinctive congestion patterns in <matrix_of_distinctive_cong_pattern>
df = pd.read_csv(dir_of_saved_ndarray_files + 'label_of_dist_feasible_samples___basis_vector_labeling.csv', sep=',',header=None)
labels_of_distinctive_congestion_patterns___basis_vector_labeling = df.values
#
# array of number of samples that has the same congestion pattern as each distinctive congestion patterns in <matrix_of_distinctive_cong_pattern>
df = pd.read_csv(dir_of_saved_ndarray_files + 'array_of_num_of_patterns_for_each_dist_cong_pttrn.csv', sep=',',header=None)
array_of_num_of_patterns_for_each_dist_cong_pttrn = [i[0] for i in df.values]
#
# matrix of all infeasible samples!
df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_infeasible_samples.csv', sep=',',header=None)
All_infeasible_samples = df.values
#
df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_power_output_of_generators.csv', sep=',',header=None)
matrix_of_power_output_of_generators = df.values
#
df = pd.read_csv(dir_of_saved_ndarray_files + 'matrix_of_LMPs.csv', sep=',',header=None)
matrix_of_LMPs = df.values
#
num_of_samples_with_feasible_solution , num_of_RPPs = All_feasible_samples.shape
_ , num_of_distinctive_classes = All_labels_feasible_matrix_of___basis_vector_labeling.shape
input___num_of_branches = len(All_matrix_of_congestion_patterns_feasible[0])
_ , num_of_generators = matrix_of_power_output_of_generators.shape
_ , num_of_nodes      = matrix_of_LMPs.shape
#
sample_and_label_data = np.concatenate((All_feasible_samples, All_labels_feasible_matrix_of___basis_vector_labeling, All_labels_feasible_vector_of___single_number_labeling, matrix_of_power_output_of_generators, matrix_of_LMPs), axis = 1)
np.random.shuffle(sample_and_label_data) # Shuffles the rows
All_feasible_samples___after_shuffle                                       = sample_and_label_data[:,0:num_of_RPPs]
All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle      = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_distinctive_classes]
All_labels_feasible_vector_of___single_number_labeling___after_shuffle     = sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:num_of_RPPs + num_of_distinctive_classes + 1].reshape(len(sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:num_of_RPPs + num_of_distinctive_classes + 1]),)
All_power_output_of_generators___after_shuffle                             = sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes + 1:num_of_RPPs + num_of_distinctive_classes + 1 + num_of_generators]
All_LMPs___after_shuffle                                                   = sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes + 1 + num_of_generators:]
#
#
# normalize the data
scaler_samples = MinMaxScaler(feature_range = (0,1),copy = True)
scaled_All_feasible_samples = scaler_samples.fit_transform(All_feasible_samples___after_shuffle.reshape(-1,1)).reshape( num_of_samples_with_feasible_solution  , num_of_RPPs)
#
scaler_labels = MinMaxScaler(feature_range = (0,1),copy = True)
scaled_All_power_output_of_generators___after_shuffle = scaler_labels.fit_transform(All_power_output_of_generators___after_shuffle.reshape(-1,1)).reshape(num_of_samples_with_feasible_solution  , num_of_generators)

#
# create samples per congestion pattern
dict_sample_per_cong_pattern                     = dict()
dict_power_output_of_generators_per_cong_pattern = dict()
dict_LMPs_per_cong_pattern                       = dict()
for i in range(0, num_of_samples_with_feasible_solution):
  try:
    dict_sample_per_cong_pattern[str(int(All_labels_feasible_vector_of___single_number_labeling___after_shuffle[i]))].append(scaled_All_feasible_samples[i,:].tolist())
    dict_power_output_of_generators_per_cong_pattern[str(int(All_labels_feasible_vector_of___single_number_labeling___after_shuffle[i]))].append(scaled_All_power_output_of_generators___after_shuffle[i, :].tolist())
    dict_LMPs_per_cong_pattern[str(int(All_labels_feasible_vector_of___single_number_labeling___after_shuffle[i]))].append(All_LMPs___after_shuffle[i, :].tolist())
  except KeyError:
    dict_sample_per_cong_pattern[str(int(All_labels_feasible_vector_of___single_number_labeling___after_shuffle[i]))] = [scaled_All_feasible_samples[i,:].tolist()]
    dict_power_output_of_generators_per_cong_pattern[str(int(All_labels_feasible_vector_of___single_number_labeling___after_shuffle[i]))] = [scaled_All_power_output_of_generators___after_shuffle[i, :].tolist()]
    dict_LMPs_per_cong_pattern[str(int(All_labels_feasible_vector_of___single_number_labeling___after_shuffle[i]))] = [All_LMPs___after_shuffle[i, :].tolist()]
#
#
_ , for_model___num_of_training_samples = scaled_All_feasible_samples.shape
#
#
counter_model = 0
list_cong_patterns_that_we_train_a_model_for___basis_vector_labeling  = []
list_cong_patterns_that_we_train_a_model_for___single_number_labeling = []
for i in range(0,num_of_distinctive_classes):
  curr_distinct_cong_pattern___single_number_labeling = labels_of_distinctive_congestion_patterns___single_number_labeling[i]
  curr_distinct_cong_pattern___basis_vector_labeling  = labels_of_distinctive_congestion_patterns___basis_vector_labeling.tolist()[i]
  curr_num_of_patterns_for_curr_dist_cong_pttrn = array_of_num_of_patterns_for_each_dist_cong_pttrn[i]
  if curr_num_of_patterns_for_curr_dist_cong_pttrn == 728: # we have enough samples for training
    counter_model += 1
    list_cong_patterns_that_we_train_a_model_for___basis_vector_labeling.append(curr_distinct_cong_pattern___basis_vector_labeling)
    list_cong_patterns_that_we_train_a_model_for___single_number_labeling.append(curr_distinct_cong_pattern___single_number_labeling)
    #
    curr_RPPs_generation_samples_for_curr_cong = np.array(dict_sample_per_cong_pattern[str(curr_distinct_cong_pattern___single_number_labeling)]).reshape(curr_num_of_patterns_for_curr_dist_cong_pttrn,num_of_RPPs)
    curr_power_generation_of_generators = np.array(dict_power_output_of_generators_per_cong_pattern[str(curr_distinct_cong_pattern___single_number_labeling)]).reshape(curr_num_of_patterns_for_curr_dist_cong_pttrn,num_of_generators)
    #
    # samples_for_current_model_trarin_and_test = np.concatenate((curr_RPPs_generation_samples_for_curr_cong,np.tile(curr_distinct_cong_pattern___basis_vector_labeling,(curr_num_of_patterns_for_curr_dist_cong_pttrn,1))), axis=1)
    samples_for_current_model_trarin_and_test = curr_RPPs_generation_samples_for_curr_cong
    labels_for_current_model_trarin_and_test  = curr_power_generation_of_generators
    #
    samples_for_current_model_train, samples_for_current_model_test, labels_for_current_model_train, labels_for_current_model_test = train_test_split(samples_for_current_model_trarin_and_test, labels_for_current_model_trarin_and_test, test_size = 0.2, random_state = 42)
    #
    len_of_input_array_to_the_NN  = len(samples_for_current_model_train[0])
    len_of_output_array_of_the_NN = len(labels_for_current_model_train[0])
    #
    #
    exec("model" + str(counter_model) + """ = Sequential([
          Dense(len_of_input_array_to_the_NN, activation='relu', input_dim=len_of_input_array_to_the_NN),
          # Dense(3*len_of_input_array_to_the_NN, activation='relu'),
          # Dense(2*len_of_input_array_to_the_NN, activation='relu'),
          Dense(2*len_of_input_array_to_the_NN, activation='relu'),
          Dense(2*len_of_input_array_to_the_NN, activation='relu'),
          Dense(len_of_output_array_of_the_NN, activation='sigmoid'),
        ])
        """
         )
    #
    exec("model" + str(counter_model) + ".summary()")
    # model.summary()
    #
    # setting the parameter of the optimizer:
    tensorflow.keras.optimizers.Adam(learning_rate=0.01)
    # tensorflow.keras.optimizers.Adam()
    #
    # define the training method
    # exec("model" + str(counter_model) + ".compile(loss='mean_squared_error', optimizer='Adam')")
    exec("model" + str(counter_model) + ".compile(loss='mean_absolute_percentage_error', optimizer='Adam')")
    #
    #
    # Training the model.
    exec("model" + str(counter_model) + ".fit(samples_for_current_model_trarin_and_test, labels_for_current_model_trarin_and_test, epochs=6000, validation_split = 0.20, shuffle=True, verbose = 2, batch_size = 128)")
    #
    # validation_split: specifies the portion of the input data that is used for validation!
    # the validation data would be the last validation_split% of the scaled_All_feasible_samples, so regardless of shaffelling this ...
    # ... set remain fixed!
    #
    # testing the model
    exec("test_results_for_model" + str(counter_model) + " = model" + str(counter_model) + ".evaluate(samples_for_current_model_test, labels_for_current_model_test, batch_size=128)")
    # print test results
    exec("print('test loss, test acc:', test_results_for_model" + str(counter_model) + ")" )
    #
    # make predictions for the test data:
    exec("predictions_for_model_" + str(counter_model) + "= model" + str(counter_model) + ".predict(samples_for_current_model_test)")
    # unscaled predictions:
    exec("unscacled_predictions_for_model_" + str(counter_model) + "= scaler_labels.inverse_transform(predictions_for_model_" + str(counter_model) + ")")
    #
    #
    # unscaled true labels for the test data:
    exec("unscacled_true_labels_for_model_" + str(counter_model) + "= scaler_labels.inverse_transform(labels_for_current_model_test)")
    #

    zzzz = 5 + 6




