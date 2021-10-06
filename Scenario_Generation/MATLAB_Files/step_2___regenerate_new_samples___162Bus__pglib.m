%
%
% Programmer: Hossein Khazaei (husein.khazaei@gmail.com) 
%
%
% This code is for generating scenarios for the cases we need more samples!     
%  
%
%
mpc = pglib_opf_case162_ieee_dtc;
%
%
parent_dir = '/Users/hkhazaei/PycharmProjects/Cong_Pred__Modified_Power_Sys_New';
curr_dir = strcat(parent_dir , '/Matlab_codes');
%
%
%
%% Step 0: Load the previously generated samples!
dir_of_results_results = strcat(curr_dir , '/Results/');
%
cd(dir_of_results_results)
% read the matrix of feasible samples (random numbers generated as loads)
generated_samples___feasible_H = readmatrix('matrix_of_feasible_samples.csv');
% read the matrix_of_congestions_of_samples___feasible
matrix_of_congestions_of_samples___feasible_H = readmatrix('matrix_of_congestions_patterns___feasible.csv');
% read the label_of_feasible_samples___single_number_labeling
label_of_feasible_samples___single_number_labeling_H = readmatrix('label_of_feasible_samples___single_number_labeling.csv');
% read the label_of_feasible_samples___basis_vector_labeling
label_of_feasible_samples___basis_vector_labeling_H = readmatrix('label_of_feasible_samples___basis_vector_labeling.csv');
% read the matrix_of_distinctive_cong_pattern
matrix_of_distinctive_cong_pattern_H = readmatrix('matrix_of_distinctive_cong_pattern.csv');
% read the label_of_dist_feasible_samples___single_number_labeling
label_of_dist_feasible_samples___single_number_labeling_H = readmatrix('label_of_dist_feasible_samples___single_number_labeling.csv');
% read the label_of_dist_feasible_samples___basis_vector_labeling
label_of_dist_feasible_samples___basis_vector_labeling_H = readmatrix('label_of_dist_feasible_samples___basis_vector_labeling.csv');
% read the array_of_num_of_patterns_for_each_dist_cong_pttrn
array_of_num_of_patterns_for_each_dist_cong_pttrn_H = readmatrix('array_of_num_of_patterns_for_each_dist_cong_pttrn.csv');
% read the matrix of infeasible samples (random numbers generated as loads)
generated_samples___infeasible_H = readmatrix('matrix_of_infeasible_samples.csv');
%
% chnage the "current folder" back to the original one
cd(curr_dir)
%
%
% %% %% %% %% %% interpret the previously generated data
[num_of_previously_generated_feasible_samples , ~] = size(generated_samples___feasible_H);
[num_of_previously_generated_infeasible_samples , ~] = size(generated_samples___infeasible_H);




%% Step 1: generate random directions so that the we can search on these directions!




%% Step 1: remove the start-up / ramp costs of generators and other irrelevant data of generators 
[mpc.H_num_of_gen , ~] = size(mpc.gen);
%
% set max/min capcacity of generators to infinity:
mpc.gen(:,9)  = inf*ones(mpc.H_num_of_gen,1);
mpc.gen(:,10) = 0*ones(mpc.H_num_of_gen,1);
%
mpc.gen(:,11) = mpc.gen(:,10);
mpc.gen(:,12) = mpc.gen(:,9);
%
mpc.gen(:,13) = mpc.gen(:,10);
mpc.gen(:,14) = mpc.gen(:,9);
%
mpc.gen(:,15) = mpc.gen(:,10);
mpc.gen(:,16) = mpc.gen(:,9);
%
mpc.gen(:,17) = inf*ones(mpc.H_num_of_gen,1);
mpc.gen(:,18) = inf*ones(mpc.H_num_of_gen,1);
%
mpc.gen(:,19) = inf*ones(mpc.H_num_of_gen,1);
mpc.gen(:,20) = inf*ones(mpc.H_num_of_gen,1);
%
mpc.gen(:,21) = 0*ones(mpc.H_num_of_gen,1);
mpc.gen(:,22) = 0*ones(mpc.H_num_of_gen,1);
mpc.gen(:,23) = 0*ones(mpc.H_num_of_gen,1);
mpc.gen(:,24) = 0*ones(mpc.H_num_of_gen,1);
mpc.gen(:,25) = 0*ones(mpc.H_num_of_gen,1);
%
%
%% Step 2: fix the data of branches to be able to use them for DC-OPF 
[mpc.H_num_of_branch , ~] = size(mpc.branch);
%
mpc.branch(:,3)  = 0*ones(mpc.H_num_of_branch,1); % R of lines 
mpc.branch(:,5)  = 0*ones(mpc.H_num_of_branch,1); % B of lines 
mpc.branch(:,9)  = 0*ones(mpc.H_num_of_branch,1); % TAP of lines 
mpc.branch(:,10) = 0*ones(mpc.H_num_of_branch,1); % SHIFT of lines 
%
%
%% Step 3: Decide the nodes to set the random demand on them
% Here, since we already have fixed loads on the buses, we just add some noise to them      
mpc.H_num_of_nodes = max([mpc.branch(:,1);mpc.branch(:,2)]);
%
mpc.H_list_of_buses_to_add_noise_to_load = [125, 62, 72, 8, 124, 14, 103, 3, 27, 60, 126, 10, 52, 147, 13, 12, 123, 94, 132, 98, 148, 96, 30, 95, 93, 117, 50, 59, 93, 117, 50, 59, 15, 78, 17, 31, 54];
%
% mpc.H_list_of_std_of_noise_of_loads = [-0.05 * abs(mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3)) , 0.15 * abs(mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3))];
mpc.H_list_of_std_of_noise_of_loads = [ linspace(-0.05, 0.3, length(mpc.H_list_of_buses_to_add_noise_to_load))' .* abs(mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3))    ,     linspace(0.05, 0.6, length(mpc.H_list_of_buses_to_add_noise_to_load))' .* abs(mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3))];
%
%
%