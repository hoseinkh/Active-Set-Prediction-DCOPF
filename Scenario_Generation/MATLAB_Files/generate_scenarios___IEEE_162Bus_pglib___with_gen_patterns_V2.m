%
%
% Programmer: Hossein Khazaei (husein.khazaei@gmail.com) 
% this is the one with Gaussian samples.
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
% num_of_samples = 20000;
num_of_samples = 40000;
num_RPPs = 37;
%
% Here, instead of using MATPOWER standard libraries, we use the The Power  
% ... Grid Library for Benchmarking AC Optimal Power Flow Algorithms!
% ... for more info, see ==> https://arxiv.org/abs/1908.02788

% mpc = pglib_opf_case118_ieee;
% mpc = pglib_opf_case179_goc; % this case is faulty (has 180 nodes instead of 179!)
% mpc = pglib_opf_case200_tamu; % too high line capacitie comparing to the nodal loads! 
mpc = pglib_opf_case162_ieee_dtc;
% mpc = loadcase('case57');
%
% curr_dir = pwd;
parent_dir = '/Users/h.khazaei/AppsH/PycharmProjects/Cong_Pred__Modified_Power_Sys_New/Codes';
curr_dir = strcat(parent_dir , '/Matlab_codes');
%
%
%% Step 1: remove the start-up / ramp costs of generators and other irrelevant data of generators 
[mpc.H_num_of_gen , ~] = size(mpc.gen);
%
% set max/min capcacity of generators to infinity:
% mpc.gen(:,9)  = inf*ones(mpc.H_num_of_gen,1);
% mpc.gen(:,10) = 0*ones(mpc.H_num_of_gen,1);
% mpc.gen(:,10) = -inf*ones(mpc.H_num_of_gen,1);
%
% make the negative loads become positive
% mpc.bus(:,3) = abs(mpc.bus(:,3));
%
% setting the Q, Qmin, and Qmax to zero (we don't need them for DC-OPF)
mpc.gen(:,3) = 0;
mpc.gen(:,4) = 0;
mpc.gen(:,5) = 0;
% setting some of the other parameters for the generator (see page 150 of Matpower Manual)
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
%% Step 2: fix the data of branches to be able to use them for DC-OPF 
[mpc.H_num_of_branch , ~] = size(mpc.branch);
%
mpc.branch(:,3)  = 0*ones(mpc.H_num_of_branch,1); % R of lines 
mpc.branch(:,5)  = 0*ones(mpc.H_num_of_branch,1); % B of lines 
mpc.branch(:,9)  = 0*ones(mpc.H_num_of_branch,1); % TAP of lines 
mpc.branch(:,10) = 0*ones(mpc.H_num_of_branch,1); % SHIFT of lines 
%
% set the short-term and emergency capacity of lines equal to long-term capacity 
mpc.branch(:,7) = mpc.branch(:,6);
mpc.branch(:,8) = mpc.branch(:,6);
%
%% Step 3: Decide the nodes to set the random demand on them
% Here, since we already have fixed loads on the buses, we just add some noise to them      
mpc.H_num_of_nodes = max([mpc.branch(:,1);mpc.branch(:,2)]);
%
% mpc.H_list_of_buses_to_add_noise_to_load = [62,124,126,17,125,72,8,14,27,30,103,60,   140,  150,  160];
% mpc.H_list_of_buses_to_add_noise_to_load = [125, 62, 72, 8, 124, 14, 103, 3, 27, 60, 126, 10, 52, 147, 13, 12, 123, 94, 132, 98, 148, 96, 30, 95, 93, 117, 50, 59, 93, 117, 50, 59, 15, 78, 17, 31, 54];
% mpc.H_list_of_buses_to_add_noise_to_load = [125, 72, 8, 14, 103, 3, 27, 60, 10, 52, 147, 13, 12, 123, 94, 132, 98, 148, 96, 30, 95, 93, 117, 50, 59, 93, 117, 50, 59, 78, 31, 54];
%
rng(65,'twister');
s = rng;
rng(s);
% mpc.H_list_of_buses_to_add_noise_to_load = randi(162,1,num_RPPs);
mpc.H_list_of_buses_to_add_noise_to_load = [5,6,9,17,17,18,18,18,22,27,27,28,30,36,36,38,39,47,50,57,73,74,79,82,84,87,90,91,103,108,111,114,120,125,129,151,160];
%
% to increase the number of feasible cases
% original_loads_at__buses_to_add_noise = 1.5*(mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3));
%
% % % Here we add a Normal noise to the loads specified in mpc.H_list_of_buses_to_add_noise_to_load 
% mpc.H_mean_of_noises_to_add_noise_to_load = 1.5*abs(mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3));
coeff_1 = 0.5;
coeff_2 = 0.4;
mpc.H_mean_of_noises_to_add_noise_to_load = coeff_1*sum(mpc.bus(:,3))/num_RPPs * ones(1,num_RPPs);
mpc.H_std_of_noises_to_add_noise_to_load  = coeff_2 .* mpc.H_mean_of_noises_to_add_noise_to_load';
% mpc.H_mean_of_noises_to_add_noise_to_load = coeff_1*mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3);
% mpc.H_std_of_noises_to_add_noise_to_load  = coeff_2 .* mpc.H_mean_of_noises_to_add_noise_to_load';
% we assume that the noises are independent
mpc.H_cov_matrix_of_noises_to_add_noise_to_load = diag(mpc.H_std_of_noises_to_add_noise_to_load);
%
%
original_loads_at__buses_to_add_noise = mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3);
%% step 4: generate samples
%
% The following command is for the uniform pdf of the noises!
% generated_samples___ALL = zeros(num_of_samples,length(mpc.H_list_of_buses_to_add_noise_to_load));
% for i = 1:length(mpc.H_list_of_buses_to_add_noise_to_load)
%     curr_matrix_of_noises = unifrnd( 0 - mpc.H_list_of_std_of_noise_of_loads(i,1) ,  0 + mpc.H_list_of_std_of_noise_of_loads(i,2) ,  num_of_samples , 1);
%     generated_samples___ALL(:,i) = curr_matrix_of_noises;
% end
% % The following command is for the Normal pdf of the noises!
generated_samples___ALL = mvnrnd(mpc.H_mean_of_noises_to_add_noise_to_load,mpc.H_cov_matrix_of_noises_to_add_noise_to_load,num_of_samples);
%
list_of_indices_of_samples_with_feasible_solutions = zeros(1,num_of_samples);
%
matrix_of_feasible_branch_and_gen_patterns_of_samples___ALL  = zeros(num_of_samples,mpc.H_num_of_branch+mpc.H_num_of_gen);
matrix_of_feasible_congestions_of_samples___ALL  = zeros(num_of_samples,mpc.H_num_of_branch);
matrix_of_feasible_gen_patterns_of_samples___ALL = zeros(num_of_samples,mpc.H_num_of_gen);
% saving the data for distinctive cong pttrns
result_struct = struct;
result_struct.num_of_distinctive_branch_and_gen_patterns = 0;
result_struct.num_of_distinctive_cong_patterns = 0;
result_struct.num_of_distinctive_gen_patterns = 0;
%
result_struct.matrix_of_distinctive_cong_pattern = []; % each row is a distinctive cong pattern 
result_struct.matrix_of_distinctive_gen_pattern = []; % each row is a distinctive gen pattern 
result_struct.matrix_of_distinctive_branch_and_gen_pattern = []; % each row is a distinctive cong pattern 
%
result_struct.array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn = [];  
result_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn = [];  
result_struct.array_of_num_of_patterns_for_each_dist_gen_pttrn  = [];  
%
result_struct.num_of_samples_with_feasible_solutions = 0;
result_struct.matrix_of_congestions_of_samples___feasible = []; % will be filled later  
result_struct.matrix_of_gen_patterns_of_samples___feasible = []; % will be filled later 
result_struct.matrix_of_branch_and_gen_patterns_of_samples___feasible = []; % will be filled later  
%
result_struct.label_of_feasible_samples___single_number_labeling = []; % will be filled later    
result_struct.label_of_dist_feasible_samples___single_number_labeling = []; % will be filled later   
result_struct.label_of_dist_feasible_samples___basis_vector_labeling =  []; % will be filled later   
%
result_struct.matrix_of_power_output_of_generators = []; % will be filled later   
result_struct.LMPs = []; % will be filled later   
%
% set the parameters of the DCOPF
mpopt = mpoption('verbose', 0, 'out.all', 0);
%
for i = 1:1:num_of_samples
    % set the demand at the buses that we generated sample for
    mpc.bus(mpc.H_list_of_buses_to_add_noise_to_load,3) = original_loads_at__buses_to_add_noise - generated_samples___ALL(i,:)';
    %
    %
    % run DCOPF and see which lines are congested!
    %
    results = rundcopf(mpc, mpopt);
    %
    %
    if results.success == 0 % infeasible solution
        fprintf('The percentage of progress is %3.2f --- Infeasible sample \n',i*100/num_of_samples)
        fprintf('The percentage of progress is %3.2f --- Infeasible sample \n',i*100/num_of_samples)
        fprintf('The percentage of progress is %3.2f --- Infeasible sample \n',i*100/num_of_samples)
        fprintf('The percentage of progress is %3.2f --- Infeasible sample \n',i*100/num_of_samples)
        fprintf('The percentage of progress is %3.2f --- Infeasible sample \n',i*100/num_of_samples)
        fprintf('The percentage of progress is %3.2f --- Infeasible sample \n',i*100/num_of_samples)
        fprintf('The percentage of progress is %3.2f --- Infeasible sample \n',i*100/num_of_samples)
        fprintf('The percentage of progress is %3.2f --- Infeasible sample \n',i*100/num_of_samples)
        fprintf('The percentage of progress is %3.2f --- Infeasible sample \n',i*100/num_of_samples)
        continue; % do nothing
    else % sample results in feasible solution
        result_struct.num_of_samples_with_feasible_solutions = result_struct.num_of_samples_with_feasible_solutions + 1;
        list_of_indices_of_samples_with_feasible_solutions(result_struct.num_of_samples_with_feasible_solutions) = i;
        matrix_of_feasible_congestions_of_samples___ALL(i,:) = (results.branch(:,18) > 0.00001*ones(mpc.H_num_of_branch,1)) + 2*(results.branch(:,19) > 0.00001*ones(mpc.H_num_of_branch,1));  
        matrix_of_feasible_gen_patterns_of_samples___ALL(i,:) = (results.gen(:,22) > 0)';
        matrix_of_feasible_branch_and_gen_patterns_of_samples___ALL(i,:) = [matrix_of_feasible_congestions_of_samples___ALL(i,:), matrix_of_feasible_gen_patterns_of_samples___ALL(i,:)];
        %
        result_struct.matrix_of_power_output_of_generators = [result_struct.matrix_of_power_output_of_generators;results.gen(:,2)'];
        result_struct.LMPs = [result_struct.LMPs;results.bus(:,14)'];
        %
        %% the following is used later on for saving the data for distinctive beanch and gen patterns!  
        %
        if result_struct.num_of_distinctive_branch_and_gen_patterns == 0
            result_struct.num_of_distinctive_branch_and_gen_patterns = 1;
            result_struct.matrix_of_distinctive_branch_and_gen_pattern = matrix_of_feasible_branch_and_gen_patterns_of_samples___ALL(i,:);
            result_struct.array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn(1,1) = 1;
        else
            is_branch_and_gen_pttrn_new = true;
            for j = 1:result_struct.num_of_distinctive_branch_and_gen_patterns
                if sum(abs(   matrix_of_feasible_branch_and_gen_patterns_of_samples___ALL(i,:)   -   result_struct.matrix_of_distinctive_branch_and_gen_pattern(j,:)  )) <= 0.0001
                    is_branch_and_gen_pttrn_new = false;
                    result_struct.array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn(j,1) = result_struct.array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn(j,1) + 1;
                    break
                else
                end
            end
            %
            if is_branch_and_gen_pttrn_new
                result_struct.num_of_distinctive_branch_and_gen_patterns = result_struct.num_of_distinctive_branch_and_gen_patterns + 1;
                result_struct.matrix_of_distinctive_branch_and_gen_pattern(result_struct.num_of_distinctive_branch_and_gen_patterns,:) = matrix_of_feasible_branch_and_gen_patterns_of_samples___ALL(i,:);
                result_struct.array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn(result_struct.num_of_distinctive_branch_and_gen_patterns,1) = 1;
            else
            end
        end
        %
        %
        %% the following is used later on for saving the data for distinctive cong patterns!  
        %
        if result_struct.num_of_distinctive_cong_patterns == 0
            result_struct.num_of_distinctive_cong_patterns = 1;
            result_struct.matrix_of_distinctive_cong_pattern = matrix_of_feasible_congestions_of_samples___ALL(i,:);
            result_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn(1,1) = 1;
        else
            is_cong_pttrn_new = true;
            for j = 1:result_struct.num_of_distinctive_cong_patterns
                if sum(abs(   matrix_of_feasible_congestions_of_samples___ALL(i,:)   -   result_struct.matrix_of_distinctive_cong_pattern(j,:)  )) <= 0.0001
                    is_cong_pttrn_new = false;
                    result_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn(j,1) = result_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn(j,1) + 1;
                    break
                else
                end
            end
            %
            if is_cong_pttrn_new
                result_struct.num_of_distinctive_cong_patterns = result_struct.num_of_distinctive_cong_patterns + 1;
                result_struct.matrix_of_distinctive_cong_pattern(result_struct.num_of_distinctive_cong_patterns,:) = matrix_of_feasible_congestions_of_samples___ALL(i,:);
                result_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn(result_struct.num_of_distinctive_cong_patterns,1) = 1;
            else
            end
        end
        %
        %
        %% the following is used later on for saving the data for distinctive gen patterns!  
        if result_struct.num_of_distinctive_gen_patterns == 0
            result_struct.num_of_distinctive_gen_patterns = 1;
            result_struct.matrix_of_distinctive_gen_pattern = matrix_of_feasible_gen_patterns_of_samples___ALL(i,:);
            result_struct.array_of_num_of_patterns_for_each_dist_gen_pttrn(1,1) = 1;
        else
            is_gen_pttrn_new = true;
            for j = 1:result_struct.num_of_distinctive_gen_patterns
                if sum(abs(   matrix_of_feasible_gen_patterns_of_samples___ALL(i,:)   -   result_struct.matrix_of_distinctive_gen_pattern(j,:)  )) <= 0.0001
                    is_gen_pttrn_new = false;
                    result_struct.array_of_num_of_patterns_for_each_dist_gen_pttrn(j,1) = result_struct.array_of_num_of_patterns_for_each_dist_gen_pttrn(j,1) + 1;
                    break
                else
                end
            end
            %
            if is_gen_pttrn_new
                result_struct.num_of_distinctive_gen_patterns = result_struct.num_of_distinctive_gen_patterns + 1;
                result_struct.matrix_of_distinctive_gen_pattern(result_struct.num_of_distinctive_gen_patterns,:) = matrix_of_feasible_gen_patterns_of_samples___ALL(i,:);
                result_struct.array_of_num_of_patterns_for_each_dist_gen_pttrn(result_struct.num_of_distinctive_gen_patterns,1) = 1;
            else
            end
        end
        
        
        
        %
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
list_of_indices_of_samples_with_feasible_solutions = list_of_indices_of_samples_with_feasible_solutions(1:result_struct.num_of_samples_with_feasible_solutions);
generated_samples___feasible = generated_samples___ALL(list_of_indices_of_samples_with_feasible_solutions,:);
% result_struct.matrix_of_congestions_of_samples___feasible = matrix_of_feasible_congestions_of_samples___ALL(list_of_indices_of_samples_with_feasible_solutions,:);
result_struct.matrix_of_branch_and_gen_patterns_of_samples___feasible = matrix_of_feasible_branch_and_gen_patterns_of_samples___ALL(list_of_indices_of_samples_with_feasible_solutions,:);
%
result_struct.label_of_dist_feasible_samples___single_number_labeling = [1:1:result_struct.num_of_distinctive_branch_and_gen_patterns];
result_struct.label_of_dist_feasible_samples___basis_vector_labeling  = eye(result_struct.num_of_distinctive_branch_and_gen_patterns);
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
result_struct.label_of_feasible_samples___single_number_labeling = zeros(result_struct.num_of_samples_with_feasible_solutions,1);
%
for i = 1:result_struct.num_of_samples_with_feasible_solutions
    for j = 1:result_struct.num_of_distinctive_branch_and_gen_patterns
        if sum(abs( result_struct.matrix_of_distinctive_branch_and_gen_pattern(j,:) - result_struct.matrix_of_branch_and_gen_patterns_of_samples___feasible(i,:) ) ) <= 0.0001
            result_struct.label_of_feasible_samples___single_number_labeling(i,1) = j;
            break
        else
        end
    end
end
%
%
result_struct.label_of_feasible_samples___basis_vector_labeling = zeros(result_struct.num_of_samples_with_feasible_solutions,result_struct.num_of_distinctive_branch_and_gen_patterns);
%
for i = 1:result_struct.num_of_samples_with_feasible_solutions
    result_struct.label_of_feasible_samples___basis_vector_labeling(i,result_struct.label_of_feasible_samples___single_number_labeling(i,1)) = 1;
end
%
%
%
%
%% Step 6: save files in the folder
if false
    dir_of_results = strcat(curr_dir , '/Results');
    % save the matrix of feasible samples (random numbers generated as loads)
    save(strcat( dir_of_results , '/matrix_of_feasible_samples' ),'generated_samples___feasible');
    % save the matrix_of_branch_and_gen_patterns_of_samples___feasible
    matrix_of_branch_and_gen_patterns_of_samples___feasible_H = result_struct.matrix_of_branch_and_gen_patterns_of_samples___feasible;
    save(strcat( dir_of_results , '/matrix_of_branch_and_gen_patterns___feasible'),'matrix_of_branch_and_gen_patterns_of_samples___feasible_H');
    % save the label_of_feasible_samples___single_number_labeling
    label_of_feasible_samples___single_number_labeling_H = result_struct.label_of_feasible_samples___single_number_labeling;
    save(strcat( dir_of_results , '/label_of_feasible_samples___single_number_labeling'),'label_of_feasible_samples___single_number_labeling_H');
    % save the label_of_feasible_samples___basis_vector_labeling
    label_of_feasible_samples___basis_vector_labeling_H = result_struct.label_of_feasible_samples___basis_vector_labeling;
    save(strcat( dir_of_results , '/label_of_feasible_samples___basis_vector_labeling'),'label_of_feasible_samples___basis_vector_labeling_H');
    % save the matrix_of_distinctive_branch_and_gen_pattern
    matrix_of_distinctive_branch_and_gen_pattern_H = result_struct.matrix_of_distinctive_branch_and_gen_pattern;
    save(strcat( dir_of_results , '/matrix_of_distinctive_branch_and_gen_pattern' ),'matrix_of_distinctive_branch_and_gen_pattern_H');
    % save the matrix_of_distinctive_cong_pattern
    matrix_of_distinctive_cong_pattern_H = result_struct.matrix_of_distinctive_cong_pattern;
    save(strcat( dir_of_results , '/matrix_of_distinctive_cong_pattern' ),'matrix_of_distinctive_cong_pattern_H');
    % save the matrix_of_distinctive_gen_pattern
    matrix_of_distinctive_gen_pattern_H = result_struct.matrix_of_distinctive_gen_pattern;
    save(strcat( dir_of_results , '/matrix_of_distinctive_gen_pattern' ),'matrix_of_distinctive_gen_pattern_H');
    % save the label_of_dist_feasible_samples___single_number_labeling
    label_of_dist_feasible_samples___single_number_labeling_H = result_struct.label_of_dist_feasible_samples___single_number_labeling;
    save(strcat(dir_of_results , '/label_of_dist_feasible_samples___single_number_labeling'),'label_of_dist_feasible_samples___single_number_labeling_H');
    % save the label_of_dist_feasible_samples___basis_vector_labeling
    label_of_dist_feasible_samples___basis_vector_labeling_H = result_struct.label_of_dist_feasible_samples___basis_vector_labeling;
    save(strcat(dir_of_results , '/label_of_dist_feasible_samples___basis_vector_labeling'),'label_of_dist_feasible_samples___basis_vector_labeling_H');
    % save the array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn
    array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn_H = result_struct.array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn;
    save(strcat( dir_of_results , '/array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn' ),'array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn_H');
    % save the array_of_num_of_patterns_for_each_dist_cong_pttrn
    array_of_num_of_patterns_for_each_dist_cong_pttrn_H = result_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn;
    save(strcat( dir_of_results , '/array_of_num_of_patterns_for_each_dist_cong_pttrn' ),'array_of_num_of_patterns_for_each_dist_cong_pttrn_H');
    % save the array_of_num_of_patterns_for_each_dist_gen_pttrn
    array_of_num_of_patterns_for_each_dist_gen_pttrn_H = result_struct.array_of_num_of_patterns_for_each_dist_gen_pttrn;
    save(strcat( dir_of_results , '/array_of_num_of_patterns_for_each_dist_gen_pttrn' ),'array_of_num_of_patterns_for_each_dist_gen_pttrn_H');
    % save the matrix of infeasible samples (random numbers generated as loads)
    save(strcat( dir_of_results , '/matrix_of_infeasible_samples' ),'generated_samples___infeasible');
    % save the matrix of power generation of generators at optimal solution
    matrix_of_power_output_of_generators_H = result_struct.matrix_of_power_output_of_generators;
    save(strcat( dir_of_results , '/matrix_of_power_output_of_generators' ),'matrix_of_power_output_of_generators');
    % save the matrix of LMPs
    matrix_of_LMPs_H = result_struct.LMPs;
    save(strcat( dir_of_results , '/matrix_of_LMPs' ),'matrix_of_LMPs_H');
%
%
elseif true
    % change the "current folder" to the one that contains the results!
    dir_of_results_results = strcat(curr_dir , '/Results/');
    cd(dir_of_results_results)
    % save the matrix of feasible samples (random numbers generated as loads)
    writematrix( generated_samples___feasible , 'matrix_of_feasible_samples.csv');
    % save the matrix_of_branch_and_gen_patterns_of_samples___feasible
    matrix_of_branch_and_gen_patterns_of_samples___feasible_H = result_struct.matrix_of_branch_and_gen_patterns_of_samples___feasible;
    writematrix( matrix_of_branch_and_gen_patterns_of_samples___feasible_H , 'matrix_of_branch_and_gen_patterns_of_samples___feasible.csv');
    % save the label_of_feasible_samples___single_number_labeling
    label_of_feasible_samples___single_number_labeling_H = result_struct.label_of_feasible_samples___single_number_labeling;
    writematrix( label_of_feasible_samples___single_number_labeling_H , 'label_of_feasible_samples___single_number_labeling.csv');
    % save the label_of_feasible_samples___basis_vector_labeling
    label_of_feasible_samples___basis_vector_labeling_H = result_struct.label_of_feasible_samples___basis_vector_labeling;
    writematrix( label_of_feasible_samples___basis_vector_labeling_H , 'label_of_feasible_samples___basis_vector_labeling.csv');
    % save the matrix_of_distinctive_branch_and_gen_pattern
    matrix_of_distinctive_branch_and_gen_pattern_H = result_struct.matrix_of_distinctive_branch_and_gen_pattern;
    writematrix( matrix_of_distinctive_branch_and_gen_pattern_H , 'matrix_of_distinctive_branch_and_gen_pattern.csv');
    % save the matrix_of_distinctive_cong_pattern
    matrix_of_distinctive_cong_pattern_H = result_struct.matrix_of_distinctive_cong_pattern;
    writematrix( matrix_of_distinctive_cong_pattern_H , 'matrix_of_distinctive_cong_pattern.csv');
    % save the matrix_of_distinctive_gen_pattern
    matrix_of_distinctive_gen_pattern_H = result_struct.matrix_of_distinctive_gen_pattern;
    writematrix( matrix_of_distinctive_gen_pattern_H , 'matrix_of_distinctive_gen_pattern.csv');
    % save the label_of_dist_feasible_samples___single_number_labeling
    label_of_dist_feasible_samples___single_number_labeling_H = result_struct.label_of_dist_feasible_samples___single_number_labeling;
    writematrix( label_of_dist_feasible_samples___single_number_labeling_H , 'label_of_dist_feasible_samples___single_number_labeling.csv');
    % save the label_of_dist_feasible_samples___basis_vector_labeling
    label_of_dist_feasible_samples___basis_vector_labeling_H = result_struct.label_of_dist_feasible_samples___basis_vector_labeling;
    writematrix( label_of_dist_feasible_samples___basis_vector_labeling_H , 'label_of_dist_feasible_samples___basis_vector_labeling.csv');
    % save the array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn
    array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn_H = result_struct.array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn;
    writematrix( array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn_H , 'array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn.csv');
    % save the array_of_num_of_patterns_for_each_dist_cong_pttrn
    array_of_num_of_patterns_for_each_dist_cong_pttrn_H = result_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn;
    writematrix( array_of_num_of_patterns_for_each_dist_cong_pttrn_H , 'array_of_num_of_patterns_for_each_dist_cong_pttrn.csv');
    % save the array_of_num_of_patterns_for_each_dist_gen_pttrn
    array_of_num_of_patterns_for_each_dist_gen_pttrn_H = result_struct.array_of_num_of_patterns_for_each_dist_gen_pttrn;
    writematrix( array_of_num_of_patterns_for_each_dist_gen_pttrn_H , 'array_of_num_of_patterns_for_each_dist_gen_pttrn.csv');
    % save the matrix of infeasible samples (random numbers generated as loads)
    writematrix( generated_samples___infeasible , 'matrix_of_infeasible_samples.csv');
    % save the matrix of power generation of generators at optimal solution
    matrix_of_power_output_of_generators_H = result_struct.matrix_of_power_output_of_generators;
    writematrix( matrix_of_power_output_of_generators_H , 'matrix_of_power_output_of_generators.csv');
    % save the matrix of LMPs
    matrix_of_LMPs_H = result_struct.LMPs;
    writematrix( matrix_of_LMPs_H , 'matrix_of_LMPs.csv');
    %
    % chnage the "current folder" back to the original one
    cd(curr_dir)
    %
    % %% %% %% % now save the CSV files in the corresponding Python folder!
    % change the "current folder" to the one that contains the results!
    dir_of_results_results_for_python = strcat(parent_dir , '/Data/');
    cd(dir_of_results_results_for_python)
    % save the matrix of feasible samples (random numbers generated as loads)
    writematrix( generated_samples___feasible , 'matrix_of_feasible_samples.csv');
    % save the matrix_of_branch_and_gen_patterns_of_samples___feasible
    matrix_of_branch_and_gen_patterns_of_samples___feasible_H = result_struct.matrix_of_branch_and_gen_patterns_of_samples___feasible;
    writematrix( matrix_of_branch_and_gen_patterns_of_samples___feasible_H , 'matrix_of_branch_and_gen_patterns_of_samples___feasible.csv');
    % save the label_of_feasible_samples___single_number_labeling
    label_of_feasible_samples___single_number_labeling_H = result_struct.label_of_feasible_samples___single_number_labeling;
    writematrix( label_of_feasible_samples___single_number_labeling_H , 'label_of_feasible_samples___single_number_labeling.csv');
    % save the label_of_feasible_samples___basis_vector_labeling
    label_of_feasible_samples___basis_vector_labeling_H = result_struct.label_of_feasible_samples___basis_vector_labeling;
    writematrix( label_of_feasible_samples___basis_vector_labeling_H , 'label_of_feasible_samples___basis_vector_labeling.csv');
    % save the matrix_of_distinctive_branch_and_gen_pattern
    matrix_of_distinctive_branch_and_gen_pattern_H = result_struct.matrix_of_distinctive_branch_and_gen_pattern;
    writematrix( matrix_of_distinctive_branch_and_gen_pattern_H , 'matrix_of_distinctive_branch_and_gen_pattern.csv');
    % save the matrix_of_distinctive_cong_pattern
    matrix_of_distinctive_cong_pattern_H = result_struct.matrix_of_distinctive_cong_pattern;
    writematrix( matrix_of_distinctive_cong_pattern_H , 'matrix_of_distinctive_cong_pattern.csv');
    % save the matrix_of_distinctive_gen_pattern
    matrix_of_distinctive_gen_pattern_H = result_struct.matrix_of_distinctive_gen_pattern;
    writematrix( matrix_of_distinctive_gen_pattern_H , 'matrix_of_distinctive_gen_pattern.csv');
    % save the label_of_dist_feasible_samples___single_number_labeling
    label_of_dist_feasible_samples___single_number_labeling_H = result_struct.label_of_dist_feasible_samples___single_number_labeling;
    writematrix( label_of_dist_feasible_samples___single_number_labeling_H , 'label_of_dist_feasible_samples___single_number_labeling.csv');
    % save the label_of_dist_feasible_samples___basis_vector_labeling
    label_of_dist_feasible_samples___basis_vector_labeling_H = result_struct.label_of_dist_feasible_samples___basis_vector_labeling;
    writematrix( label_of_dist_feasible_samples___basis_vector_labeling_H , 'label_of_dist_feasible_samples___basis_vector_labeling.csv');
    % save the array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn
    array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn_H = result_struct.array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn;
    writematrix( array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn_H , 'array_of_num_of_patterns_for_each_dist_branch_and_gen_pttrn.csv');
    % save the array_of_num_of_patterns_for_each_dist_cong_pttrn
    array_of_num_of_patterns_for_each_dist_cong_pttrn_H = result_struct.array_of_num_of_patterns_for_each_dist_cong_pttrn;
    writematrix( array_of_num_of_patterns_for_each_dist_cong_pttrn_H , 'array_of_num_of_patterns_for_each_dist_cong_pttrn.csv');
    % save the array_of_num_of_patterns_for_each_dist_gen_pttrn
    array_of_num_of_patterns_for_each_dist_gen_pttrn_H = result_struct.array_of_num_of_patterns_for_each_dist_gen_pttrn;
    writematrix( array_of_num_of_patterns_for_each_dist_gen_pttrn_H , 'array_of_num_of_patterns_for_each_dist_gen_pttrn.csv');
    % save the matrix of infeasible samples (random numbers generated as loads)
    writematrix( generated_samples___infeasible , 'matrix_of_infeasible_samples.csv');
    % save the matrix of power generation of generators at optimal solution
    matrix_of_power_output_of_generators_H = result_struct.matrix_of_power_output_of_generators;
    writematrix( matrix_of_power_output_of_generators_H , 'matrix_of_power_output_of_generators.csv');
    % save the matrix of LMPs
    matrix_of_LMPs_H = result_struct.LMPs;
    writematrix( matrix_of_LMPs_H , 'matrix_of_LMPs.csv');
    %
    % chnage the "current folder" back to the original one
    cd(curr_dir)
    %
end
%
%
%
5+6






