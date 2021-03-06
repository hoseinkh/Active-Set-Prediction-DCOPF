#
#
#
# This code will read the MAT files generated by MATLAB and saved at directory ./Matlab_codes/Results, convert them ...
# ... into ndarray and saves them in the directory ./Data
#
# These files will be used by other programs.
#
# Note that here the focus is on reading the scenarios generated by MATLAB. For more info, see the MATLAB code named: generate_scenarios.m
#
#
# Here I load and save the data in the same order as they have been saved in MATLAB
#
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#
import scipy.io
import os
import numpy as np
# get the current working directory:
curr_dir = os.getcwd()
dir_of_MAT_files = curr_dir + '/Matlab_codes/Results/'
dir_to_save_ndarrays = curr_dir + '/Data/'
#
# load and save "matrix_of_feasible_samples"
mat = scipy.io.loadmat(dir_of_MAT_files + 'matrix_of_feasible_samples.mat')
matrix_of_feasible_samples = mat['generated_samples___feasible']
np.savetxt(dir_to_save_ndarrays + 'matrix_of_feasible_samples.csv', matrix_of_feasible_samples, delimiter=',')
#
# load and save "matrix_of_congestions_patterns___feasible"
mat = scipy.io.loadmat(dir_of_MAT_files + 'matrix_of_congestions_patterns___feasible.mat')
matrix_of_congestions_patterns___feasible = mat['matrix_of_congestions_of_samples___feasible_H']
np.savetxt(dir_to_save_ndarrays + 'matrix_of_congestions_patterns___feasible.csv', matrix_of_congestions_patterns___feasible, delimiter=',')
#
# load and save "label_of_feasible_samples___single_number_labeling"
mat = scipy.io.loadmat(dir_of_MAT_files + 'label_of_feasible_samples___single_number_labeling.mat')
label_of_feasible_samples___single_number_labeling = mat['label_of_feasible_samples___single_number_labeling_H']
            # because single number labels starts from 1 in MATLAB, here we change it to start from zero!
label_of_feasible_samples___single_number_labeling = np.subtract(label_of_feasible_samples___single_number_labeling , np.ones(np.shape(label_of_feasible_samples___single_number_labeling)))
np.savetxt(dir_to_save_ndarrays + 'label_of_feasible_samples___single_number_labeling.csv', label_of_feasible_samples___single_number_labeling, delimiter=',')
#
# load and save "label_of_feasible_samples___basis_vector_labeling"
mat = scipy.io.loadmat(dir_of_MAT_files + 'label_of_feasible_samples___basis_vector_labeling.mat')
label_of_feasible_samples___basis_vector_labeling = mat['label_of_feasible_samples___basis_vector_labeling_H']
np.savetxt(dir_to_save_ndarrays + 'label_of_feasible_samples___basis_vector_labeling.csv', label_of_feasible_samples___basis_vector_labeling, delimiter=',')
#
# load and save "matrix_of_distinctive_cong_pattern"
mat = scipy.io.loadmat(dir_of_MAT_files + 'matrix_of_distinctive_cong_pattern.mat')
matrix_of_distinctive_cong_pattern = mat['matrix_of_distinctive_cong_pattern_H']
np.savetxt(dir_to_save_ndarrays + 'matrix_of_distinctive_cong_pattern.csv', matrix_of_distinctive_cong_pattern, delimiter=',')
#
#
# load and save "label_of_dist_feasible_samples___single_number_labeling"
mat = scipy.io.loadmat(dir_of_MAT_files + 'label_of_dist_feasible_samples___single_number_labeling.mat')
label_of_dist_feasible_samples___single_number_labeling = mat['label_of_dist_feasible_samples___single_number_labeling_H']
            # because single number labels starts from 1 in MATLAB, here we change it to start from zero!
label_of_dist_feasible_samples___single_number_labeling = np.subtract(label_of_dist_feasible_samples___single_number_labeling , np.ones(np.shape(label_of_dist_feasible_samples___single_number_labeling)))
np.savetxt(dir_to_save_ndarrays + 'label_of_dist_feasible_samples___single_number_labeling.csv', label_of_dist_feasible_samples___single_number_labeling, delimiter=',')
#
#
# load and save "label_of_dist_feasible_samples___basis_vector_labeling
mat = scipy.io.loadmat(dir_of_MAT_files + 'label_of_dist_feasible_samples___basis_vector_labeling.mat')
label_of_dist_feasible_samples___basis_vector_labeling = mat['label_of_dist_feasible_samples___basis_vector_labeling_H']
np.savetxt(dir_to_save_ndarrays + 'label_of_dist_feasible_samples___basis_vector_labeling.csv', label_of_dist_feasible_samples___basis_vector_labeling, delimiter=',')
#
# load and save "array_of_num_of_patterns_for_each_dist_cong_pttrn"
mat = scipy.io.loadmat(dir_of_MAT_files + 'array_of_num_of_patterns_for_each_dist_cong_pttrn.mat')
array_of_num_of_patterns_for_each_dist_cong_pttrn = mat['array_of_num_of_patterns_for_each_dist_cong_pttrn_H']
np.savetxt(dir_to_save_ndarrays + 'array_of_num_of_patterns_for_each_dist_cong_pttrn.csv', array_of_num_of_patterns_for_each_dist_cong_pttrn, delimiter=',')
#
# load and save "matrix_of_infeasible_samples"
mat = scipy.io.loadmat(dir_of_MAT_files + 'matrix_of_infeasible_samples.mat')
matrix_of_infeasible_samples = mat['generated_samples___infeasible']
np.savetxt(dir_to_save_ndarrays + 'matrix_of_infeasible_samples.csv', matrix_of_infeasible_samples, delimiter=',')
#
#
5 + 6


