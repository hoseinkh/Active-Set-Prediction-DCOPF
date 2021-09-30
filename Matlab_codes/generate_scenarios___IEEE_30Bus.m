%
%
% Programmer: Hossein Khazaei (husein.khazaei@gmail.com) 
%
%
% This code is for scenario generation and determining the congestions! 
%  
% Here, instead of creating and defining new network, we use standard ...
% ... networks available in MATPOWER, e.g. IEEE 14 bus, IEEE 30 bus, etc   
%
%
%
% Input data
num_of_samples = 20000;
%
%
mpc = loadcase('case30');
%
%
%
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
%
mpc.H_list_of_buses_to_add_noise_to_load = [2,7,8,30];
%
mpc.H_list_of_ratio_of_std_to_mean_of_noise_of_loads = [0.5,0.5,0.5,0.5]; 
% Note that we must have: length(mpc.H_list_of_buses_to_add_noise_to_load) = length(mpc.H_list_of_buses_to_add_noise_to_load)
if length(mpc.H_list_of_buses_to_add_noise_to_load) ~= length(mpc.H_list_of_buses_to_add_noise_to_load)
    fprintf("ERROR: we must have: length(mpc.H_list_of_buses_to_add_noise_to_load) = length(mpc.H_list_of_buses_to_add_noise_to_load)")
    return % this will stop the program!  
else
end
%
%
% % Here we add a Normal noise to the loads specified in mpc.H_list_of_buses_to_add_noise_to_load 
% mpc.H_mean_of_noises_to_add_noise_to_load = zeros(1,length(mpc.H_list_of_buses_to_add_noise_to_load));
mpc.H_mean_of_noises_to_add_noise_to_load = [10 , 10 , 10 , 10];
%
mpc.H_std_of_noises_to_add_noise_to_load = mpc.H_list_of_ratio_of_std_to_mean_of_noise_of_loads .* mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3)';
%
% we assume that the noises are independent
mpc.H_cov_matrix_of_noises_to_add_noise_to_load = diag(mpc.H_std_of_noises_to_add_noise_to_load);
%
%
%% step 4: generate samples
generated_samples___ALL = mvnrnd(mpc.H_mean_of_noises_to_add_noise_to_load,mpc.H_cov_matrix_of_noises_to_add_noise_to_load,num_of_samples);
%
%
original_loads_at__buses_to_add_noise = mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3);
%
list_of_indices_of_samples_with_feasible_solutions = zeros(1,num_of_samples);
%
matrix_of_feasible_congestions_of_samples___ALL = zeros(num_of_samples,mpc.H_num_of_branch);
% saving the data for distinctive cong pttrns
result_cong_pattern_struct = struct;
result_cong_pattern_struct.num_of_distinctive_cong_patterns = 0;
result_cong_pattern_struct.matrix_of_distinctive_cong_pattern = []; % each row is a distinctive cong pattern 
result_cong_pattern_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn = [];  
result_cong_pattern_struct.num_of_samples_with_feasible_solutions = 0;
result_cong_pattern_struct.matrix_of_congestions_of_samples___feasible = []; % will be filled later  
result_cong_pattern_struct.label_of_feasible_samples___single_number_labeling = []; % will be filled later    
result_cong_pattern_struct.label_of_dist_feasible_samples___single_number_labeling = []; % will be filled later   
result_cong_pattern_struct.label_of_dist_feasible_samples___basis_vector_labeling =  []; % will be filled later   
%
for i = 1:1:num_of_samples
    mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3) = original_loads_at__buses_to_add_noise + generated_samples___ALL(i,:)';
    %
    %
    % run DCOPF and see which lines are congested!
    %
    mpopt = mpoption('verbose', 0, 'out.all', 0);
    results = rundcopf(mpc, mpopt);
    %
    %
    if results.success == 0 % infeasible solution
        continue; % do nothing
    else % sample results in feasible solution
        result_cong_pattern_struct.num_of_samples_with_feasible_solutions = result_cong_pattern_struct.num_of_samples_with_feasible_solutions + 1;
        list_of_indices_of_samples_with_feasible_solutions(result_cong_pattern_struct.num_of_samples_with_feasible_solutions) = i;
        matrix_of_feasible_congestions_of_samples___ALL(i,:) = (results.branch(:,18) > 0.00001*ones(mpc.H_num_of_branch,1)) + 2*(results.branch(:,19) > 0.00001*ones(mpc.H_num_of_branch,1));  
        %
        % the following is for saving the data for distinctive cong patterns!  
        %
        if result_cong_pattern_struct.num_of_distinctive_cong_patterns == 0
            result_cong_pattern_struct.num_of_distinctive_cong_patterns = 1;
            result_cong_pattern_struct.matrix_of_distinctive_cong_pattern = matrix_of_feasible_congestions_of_samples___ALL(i,:);
            result_cong_pattern_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn(1,1) = 1;
        else
            is_cong_pttrn_new = true;
            for j = 1:result_cong_pattern_struct.num_of_distinctive_cong_patterns
                if sum(abs(   matrix_of_feasible_congestions_of_samples___ALL(i,:)   -   result_cong_pattern_struct.matrix_of_distinctive_cong_pattern(j,:)  )) <= 0.0001
                    is_cong_pttrn_new = false;
                    result_cong_pattern_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn(j,1) = result_cong_pattern_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn(j,1) + 1;
                    break
                else
                end
            end
            %
            if is_cong_pttrn_new
                result_cong_pattern_struct.num_of_distinctive_cong_patterns = result_cong_pattern_struct.num_of_distinctive_cong_patterns + 1;
                result_cong_pattern_struct.matrix_of_distinctive_cong_pattern(result_cong_pattern_struct.num_of_distinctive_cong_patterns,:) = matrix_of_feasible_congestions_of_samples___ALL(i,:);
                result_cong_pattern_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn(result_cong_pattern_struct.num_of_distinctive_cong_patterns,1) = 1;
            else
            end
        end
    end
    %
    %
    fprintf('The percentage of progress is %3.2f \n',i*100/num_of_samples)
    fprintf('The percentage of progress is %3.2f \n',i*100/num_of_samples)
    fprintf('The percentage of progress is %3.2f \n',i*100/num_of_samples)
    fprintf('The percentage of progress is %3.2f \n',i*100/num_of_samples)
    fprintf('The percentage of progress is %3.2f \n',i*100/num_of_samples)
    fprintf('The percentage of progress is %3.2f \n',i*100/num_of_samples)
    fprintf('The percentage of progress is %3.2f \n',i*100/num_of_samples)
    fprintf('The percentage of progress is %3.2f \n',i*100/num_of_samples)
    fprintf('The percentage of progress is %3.2f \n',i*100/num_of_samples)
    %
    %
end
%
%
list_of_indices_of_samples_with_feasible_solutions = list_of_indices_of_samples_with_feasible_solutions(1:result_cong_pattern_struct.num_of_samples_with_feasible_solutions);
generated_samples___feasible = generated_samples___ALL(list_of_indices_of_samples_with_feasible_solutions,:);
result_cong_pattern_struct.matrix_of_congestions_of_samples___feasible = matrix_of_feasible_congestions_of_samples___ALL(list_of_indices_of_samples_with_feasible_solutions,:);
%
result_cong_pattern_struct.label_of_dist_feasible_samples___single_number_labeling = [1:1:result_cong_pattern_struct.num_of_distinctive_cong_patterns];
result_cong_pattern_struct.label_of_dist_feasible_samples___basis_vector_labeling  = eye(result_cong_pattern_struct.num_of_distinctive_cong_patterns);
%
list_of_indices_of_samples_with_infeasible_solutions = setdiff( [1:1:num_of_samples] , list_of_indices_of_samples_with_feasible_solutions );
generated_samples___infeasible = generated_samples___ALL(list_of_indices_of_samples_with_infeasible_solutions,:);
%
%% Step 5: assigning labels to the samples.
% here, we assign two types of labels: first one is only a number ( we ...
% ... call it "single number labeling", while the second one is a ...
% ... vector which only one element is 1 and other elements are zeros ...
% ... (we call it "basis vector labeling" version)!
%
%
result_cong_pattern_struct.label_of_feasible_samples___single_number_labeling = zeros(result_cong_pattern_struct.num_of_samples_with_feasible_solutions,1);
%
for i = 1:result_cong_pattern_struct.num_of_samples_with_feasible_solutions
    for j = 1:result_cong_pattern_struct.num_of_distinctive_cong_patterns
        if sum(abs( result_cong_pattern_struct.matrix_of_distinctive_cong_pattern(j,:) - result_cong_pattern_struct.matrix_of_congestions_of_samples___feasible(i,:) ) ) <= 0.0001
            result_cong_pattern_struct.label_of_feasible_samples___single_number_labeling(i,1) = j;
            break
        else
        end
    end
end
%
%
result_cong_pattern_struct.label_of_feasible_samples___basis_vector_labeling = zeros(result_cong_pattern_struct.num_of_samples_with_feasible_solutions,result_cong_pattern_struct.num_of_distinctive_cong_patterns);
%
for i = 1:result_cong_pattern_struct.num_of_samples_with_feasible_solutions
    result_cong_pattern_struct.label_of_feasible_samples___basis_vector_labeling(i,result_cong_pattern_struct.label_of_feasible_samples___single_number_labeling(i,1)) = 1;
end
%
%
%
%
%% Step 6: save files in the folder
curr_dir = pwd;
dir_of_results = strcat(curr_dir , '/Results');
% save the matrix of feasible samples (random numbers generated as loads)
save(strcat( dir_of_results , '/matrix_of_feasible_samples' ),'generated_samples___feasible');
% save the matrix_of_congestions_of_samples___feasible
matrix_of_congestions_of_samples___feasible_H = result_cong_pattern_struct.matrix_of_congestions_of_samples___feasible;
save(strcat( dir_of_results , '/matrix_of_congestions_patterns___feasible'),'matrix_of_congestions_of_samples___feasible_H');
% save the label_of_feasible_samples___single_number_labeling
label_of_feasible_samples___single_number_labeling_H = result_cong_pattern_struct.label_of_feasible_samples___single_number_labeling;
save(strcat( dir_of_results , '/label_of_feasible_samples___single_number_labeling'),'label_of_feasible_samples___single_number_labeling_H');
% save the label_of_feasible_samples___basis_vector_labeling
label_of_feasible_samples___basis_vector_labeling_H = result_cong_pattern_struct.label_of_feasible_samples___basis_vector_labeling;
save(strcat( dir_of_results , '/label_of_feasible_samples___basis_vector_labeling'),'label_of_feasible_samples___basis_vector_labeling_H');
% save the matrix_of_distinctive_cong_pattern
matrix_of_distinctive_cong_pattern_H = result_cong_pattern_struct.matrix_of_distinctive_cong_pattern;
save(strcat( dir_of_results , '/matrix_of_distinctive_cong_pattern' ),'matrix_of_distinctive_cong_pattern_H');
% save the label_of_dist_feasible_samples___single_number_labeling
label_of_dist_feasible_samples___single_number_labeling_H = result_cong_pattern_struct.label_of_dist_feasible_samples___single_number_labeling;
save(strcat(dir_of_results , '/label_of_dist_feasible_samples___single_number_labeling'),'label_of_dist_feasible_samples___single_number_labeling_H');
% save the label_of_dist_feasible_samples___basis_vector_labeling
label_of_dist_feasible_samples___basis_vector_labeling_H = result_cong_pattern_struct.label_of_dist_feasible_samples___basis_vector_labeling;
save(strcat(dir_of_results , '/label_of_dist_feasible_samples___basis_vector_labeling'),'label_of_dist_feasible_samples___basis_vector_labeling_H');
% save the array_of_num_of_patterns_for_each_dist_cong_pttrn
array_of_num_of_patterns_for_each_dist_cong_pttrn_H = result_cong_pattern_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn;
save(strcat( dir_of_results , '/array_of_num_of_patterns_for_each_dist_cong_pttrn' ),'array_of_num_of_patterns_for_each_dist_cong_pttrn_H');
% save the matrix of infeasible samples (random numbers generated as loads)
save(strcat( dir_of_results , '/matrix_of_infeasible_samples' ),'generated_samples___infeasible');
%
%
%
5+6






