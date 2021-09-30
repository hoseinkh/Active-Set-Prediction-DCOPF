# This file trains the NN for the classification of the congestion patterns!
#
import numpy as np
import pandas as pd
#
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import StratifiedKFold
#
from sklearn.model_selection import train_test_split, GridSearchCV
#
import os
#
#
## Step 1: Reading the data of scenarios! These data are stored in the directory named <Data>, and are of .csv format!
#
#
list_of_bracnches_to_have_classifier_for = [4,148,136,10,230,205,200,133,75,82,203,222]
#
#
# curr_dir = '/Users/hkhazaei/PycharmProjects/Power_Sys_New'
# curr_dir = os.getcwd()
# curr_dir = '/Users/hkhazaei/PycharmProjects/Cong_Pred__Modified_Power_Sys_New/Codes'
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
#
num_of_samples_with_feasible_solution , num_of_RPPs = All_feasible_samples.shape
_ , num_of_distinctive_classes = All_labels_feasible_matrix_of___basis_vector_labeling.shape
num_of_branches = len(All_matrix_of_congestion_patterns_feasible[0])
#
#
## ## ##
num_of_branches_to_have_classifier_for = len(list_of_bracnches_to_have_classifier_for)
#
dic_of_branches_and_their_cong_status___no_classifier_needed = {}
for i in range(0,num_of_branches):
    if not(i in list_of_bracnches_to_have_classifier_for):
        if i in [18,176,195,227,239]:
            dic_of_branches_and_their_cong_status___no_classifier_needed[i] = 2  # Here I assumed that the default congestion for the line is 2
        else:
            dic_of_branches_and_their_cong_status___no_classifier_needed[i] = 0 # Here I assumed that the default congestion for the line is 0
        #
    else:
        pass
    #
#
#
#
sample_and_label_data = np.concatenate((All_feasible_samples, All_matrix_of_congestion_patterns_feasible, All_labels_feasible_matrix_of___basis_vector_labeling, All_labels_feasible_vector_of___single_number_labeling), axis = 1)
np.random.shuffle(sample_and_label_data) # Shuffles the rows
All_feasible_samples___after_shuffle                                       = sample_and_label_data[:,0:num_of_RPPs]
All_matrix_of_congestion_patterns_feasible___after_shuffle                 = sample_and_label_data[:,num_of_RPPs:num_of_RPPs + num_of_branches]
All_labels_feasible_matrix_of___basis_vector_labeling___after_shuffle      = sample_and_label_data[:,num_of_RPPs+ num_of_branches:num_of_RPPs + num_of_branches + num_of_distinctive_classes]
All_labels_feasible_vector_of___single_number_labeling___after_shuffle     = sample_and_label_data[:,num_of_RPPs + num_of_branches + num_of_distinctive_classes:].reshape(len(sample_and_label_data[:,num_of_RPPs + num_of_distinctive_classes:]),)
#
# normalize the data
scaler = MinMaxScaler(feature_range = (0,1),copy = True)
scaled_All_feasible_samples = scaler.fit_transform(All_feasible_samples___after_shuffle.reshape(-1,1)).reshape( num_of_samples_with_feasible_solution  , num_of_RPPs)
#
#
# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(scaled_All_feasible_samples, All_matrix_of_congestion_patterns_feasible___after_shuffle, test_size = 0.2, random_state=0)
#
num_of_samples_with_feasible_solution___train = X_train.shape[0]
num_of_samples_with_feasible_solution___test = X_test.shape[0]
# _ , for_model___num_of_training_samples = scaled_All_feasible_samples.shape
#
#
#
#
Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___train = {}
Dic_of_All_labels_feasible_basis_vector___single_line_labeling___single_number___after_shuffle___train = {}
for i in range(0,num_of_branches):
    Dic_of_All_labels_feasible_basis_vector___single_line_labeling___single_number___after_shuffle___train['clf___for_Line_' + str(int(i))] = y_train[:, int(i)]
    Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___train['clf___for_Line_' + str(int(i))] = np.zeros((num_of_samples_with_feasible_solution___train,3))
    for j in range(0,num_of_samples_with_feasible_solution___train):
        if y_train[int(j),int(i)] == 0:
            Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___train['clf___for_Line_' + str(int(i))][j,:] = np.array([1,0,0])
        elif y_train[int(j),int(i)] == 1:
            Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___train['clf___for_Line_' + str(int(i))][j,:] = np.array([0,1,0])
        elif y_train[int(j),int(i)] == 2:
            Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___train['clf___for_Line_' + str(int(i))][j,:] = np.array([0,0,1])
        else:
            pass
#
#
#
Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___test = {}
Dic_of_All_labels_feasible_basis_vector___single_line_labeling___single_number___after_shuffle___test = {}
for i in range(0,num_of_branches):
    Dic_of_All_labels_feasible_basis_vector___single_line_labeling___single_number___after_shuffle___test['clf___for_Line_' + str(int(i))] = y_test[:,int(i)]
    Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___test['clf___for_Line_' + str(int(i))] = np.zeros((num_of_samples_with_feasible_solution___test,3))
    for j in range(0,num_of_samples_with_feasible_solution___test):
        if y_test[int(j),int(i)] == 0:
            Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___test['clf___for_Line_' + str(int(i))][j,:] = np.array([1,0,0])
        elif y_test[int(j),int(i)] == 1:
            Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___test['clf___for_Line_' + str(int(i))][j,:] = np.array([0,1,0])
        elif y_test[int(j),int(i)] == 2:
            Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___test['clf___for_Line_' + str(int(i))][j,:] = np.array([0,0,1])
        else:
            pass
#
#
## Define classifers for each line:
Classifiers_ALL_dictionary = {}
for i in list_of_bracnches_to_have_classifier_for:
    # Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))] = svm.SVC(kernel='linear', C=1)
    Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))] = Sequential([
        Dense(max(num_of_branches , 150), activation='relu', input_dim=num_of_RPPs),
        Dense(2*max(num_of_branches , 150), activation='relu'),
        Dense(2*max(num_of_branches , 150), activation='relu'),
        Dense(2*max(num_of_branches , 150), activation='relu'),
        Dense(2*max(num_of_branches , 150), activation='relu'),
        Dense(3, activation='softmax'),
    ])
#
#
# setting the parameter of the optimizer:
tensorflow.keras.optimizers.Adam(learning_rate=0.00001)
#
#
#
#
for i in list_of_bracnches_to_have_classifier_for:
     # define the training method
    Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))].compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    #
    if i in [136]:
        Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))].fit(X_train,Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___train['clf___for_Line_' + str(int(i))], epochs=100, validation_split=0.01, shuffle=True, verbose=2,batch_size=40)
    elif i in [4,10,205,230]:
        Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))].fit(X_train, Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___train['clf___for_Line_' + str(int(i))], epochs=100, validation_split=0.01, shuffle=True, verbose=2, batch_size=40)
    else:
        Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))].fit(X_train, Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___train['clf___for_Line_' + str(int(i))], epochs=100, validation_split=0.01, shuffle=True, verbose=2,batch_size=40)
    #
    #
#
#
#
#
# Testing the overall performance of the code!     dic_of_branches_and_their_cong_status___no_classifier_needed
# Make predictions on unseen test data
y_predict_ALL_dictionary___basis_vector = {}
y_predict_ALL_dictionary___single_number = {}
for i in range(0,num_of_branches):
    if not (int(i) in list_of_bracnches_to_have_classifier_for):
        y_predict_ALL_dictionary___single_number['predictions___for_Line_' + str(int(i))] = int(dic_of_branches_and_their_cong_status___no_classifier_needed[i]) * np.ones((num_of_samples_with_feasible_solution___test,))
        # because in the training data we have only one type of label (congestion) for this line, in the prediction we choose the same label!
        # so no need for SVM! (This is actually SVM, but SVM algorithm implemented in Python requires at least two different of labels in the training set!)
        y_predict_ALL_dictionary___basis_vector['predictions___for_Line_' + str(int(i))] = np.ones((num_of_samples_with_feasible_solution___test,1)).reshape((num_of_samples_with_feasible_solution___test,)) * dic_of_branches_and_their_cong_status___no_classifier_needed[i]
        if int(dic_of_branches_and_their_cong_status___no_classifier_needed[i]) == 0:
            y_predict_ALL_dictionary___basis_vector['predictions___for_Line_' + str(int(i))] = np.tile([1,0,0], (num_of_samples_with_feasible_solution___test, 1))
        elif int(dic_of_branches_and_their_cong_status___no_classifier_needed[i]) == 1:
            y_predict_ALL_dictionary___basis_vector['predictions___for_Line_' + str(int(i))] = np.tile([0,1,0], (num_of_samples_with_feasible_solution___test, 1))
        elif int(dic_of_branches_and_their_cong_status___no_classifier_needed[i]) == 2:
            y_predict_ALL_dictionary___basis_vector['predictions___for_Line_' + str(int(i))] = np.tile([0, 0, 1], (num_of_samples_with_feasible_solution___test, 1))
        else:
            pass
            #
        zxzx = 5 + 6
        # number_of_missclassifies = np.count_nonzero((y_predict_ALL_dictionary['predictions___for_Line_' + str(int(i))] - Dic_of_All_labels_feasible_basis_vector___single_line_labeling___basis_vector___after_shuffle___test['clf___for_Line_' + str(int(i))]).sum(axis=1))
        # print("Accuracy of predictions for line {}: {}%".format(int(i), (num_of_samples_with_feasible_solution___test - number_of_missclassifies) * 100 / num_of_samples_with_feasible_solution___test ))
        number_of_missclassifies = np.count_nonzero(y_predict_ALL_dictionary___single_number['predictions___for_Line_' + str(int(i))] - Dic_of_All_labels_feasible_basis_vector___single_line_labeling___single_number___after_shuffle___test['clf___for_Line_' + str(int(i))])
        print("Accuracy of predictions for line {}: {}%".format(int(i), (num_of_samples_with_feasible_solution___test - number_of_missclassifies) * 100 / num_of_samples_with_feasible_solution___test))
    else:
        # y_predict_ALL_dictionary___basis_vector['predictions___for_Line_' + str(int(i))] = Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))].predict(X_test)
        y_predict_ALL_dictionary___single_number['predictions___for_Line_' + str(int(i))] = Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))].predict_classes(X_test)
        #y_predict_ALL_dictionary___basis_vector
        # _ , y_predict_ALL_dictionary___single_number['predictions___for_Line_' + str(int(i))] = np.nonzero(y_predict_ALL_dictionary___basis_vector['predictions___for_Line_' + str(int(i))])
        #
        y_predict_ALL_dictionary___basis_vector['predictions___for_Line_' + str(int(i))] = np.zeros((num_of_samples_with_feasible_solution___test,3))
        for j in range(0,num_of_samples_with_feasible_solution___test):
            if int(y_predict_ALL_dictionary___single_number['predictions___for_Line_' + str(int(i))][j]) == 0:
                y_predict_ALL_dictionary___basis_vector['predictions___for_Line_' + str(int(i))][j,:] = np.array([1,0,0])
            elif int(y_predict_ALL_dictionary___single_number['predictions___for_Line_' + str(int(i))][j]) == 1:
                y_predict_ALL_dictionary___basis_vector['predictions___for_Line_' + str(int(i))][j,:] = np.array([0,1,0])
            elif int(y_predict_ALL_dictionary___single_number['predictions___for_Line_' + str(int(i))][j]) == 2:
                y_predict_ALL_dictionary___basis_vector['predictions___for_Line_' + str(int(i))][j,:] = np.array([0,0,1])
            else:
                pass
            #
            #
        #
        #
        zxszz = 5 + 6
        number_of_missclassifies = np.count_nonzero(y_predict_ALL_dictionary___single_number['predictions___for_Line_' + str(int(i))] - Dic_of_All_labels_feasible_basis_vector___single_line_labeling___single_number___after_shuffle___test['clf___for_Line_' + str(int(i))])
        print("Accuracy of predictions for line {}: {}%  -- NN".format(int(i), (num_of_samples_with_feasible_solution___test - number_of_missclassifies) * 100 / num_of_samples_with_feasible_solution___test))
        # print("Accuracy of predictions for line {}: {}  --- NN%".format(int(i) , Classifiers_ALL_dictionary['clf___for_Line_' + str(int(i))].score(X_test, y_test[:,int(i)]) * 100 ))
#
#
# print('\n list of lines with single class label in training data = \n {}'.format(list_of_classes_with_only_one_type_of_label_in_the_training_data))
#
#
## Construct the congestion pattern
# predicted_congestion_pattern = np.array([])
for i in range(num_of_branches):
    if i == 0:
        predicted_congestion_pattern = y_predict_ALL_dictionary___single_number['predictions___for_Line_' + str(int(i))].reshape((num_of_samples_with_feasible_solution___test, 1))
    else:
        predicted_congestion_pattern = np.concatenate((predicted_congestion_pattern , y_predict_ALL_dictionary___single_number['predictions___for_Line_' + str(int(i))].reshape((num_of_samples_with_feasible_solution___test,1))) , axis=1)
#
#
## Calculate the error in congestion prediction!
difference_between_prediction_and_test_label = y_test - predicted_congestion_pattern
#
num_of_misclassified_test_samples = 0
for i in range(0,num_of_samples_with_feasible_solution___test):
    if len(np.nonzero(difference_between_prediction_and_test_label[i,:])[0]) != 0:
        num_of_misclassified_test_samples += 1
    else:
        pass
#
#
#
print('Total Forecasting Accuracy = {}%'.format((num_of_samples_with_feasible_solution___test - num_of_misclassified_test_samples) * 100 / num_of_samples_with_feasible_solution___test))

5 + 6