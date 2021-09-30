#
#
# This temp file shows how to read MAT files (MATLAB format), and save them as nparray!
#
#
## This method usually works. For details on the alternative method (when this one doesn't work) see the following link:
# Link: https://stackoverflow.com/questions/874461/read-mat-files-in-python
#
import scipy.io
mat = scipy.io.loadmat('/Users/hkhazaei/PycharmProjects/Power_Sys_New/Matlab_codes/Results/array_of_num_of_patterns_for_each_dist_cong_pttrn.mat')
#
# note that <mat> is a dictionary, and has several keys. One of the keys is the variable name we used when we ...
# ... were saving this mat file in Matlab (the code in MATLAB was: save(path, var_name_of_variable_to_be_saved)).
# This variable name can be found in the keys of the dictionary <mat>.
# if we revive the value of <mat> for that key, we get the same matrix we want, and it is saved as a ndarray matrix!
#
# here we look at the keys to find the variable name we are looking for!
print(mat.keys())
matrix_of_num_of_patterns_for_each_dist_cong_pttrn = mat['array_of_num_of_patterns_for_each_dist_cong_pttrn_H']


5 + 6