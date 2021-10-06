%% Chance Constraint DC-OPF solver
%
% vector of variables = [theta; T; P_G]
% theta: from 1 to num_of_bus
% T: from num_of_bus + 1 to num_of_bus + num_of_branch
% P_G: from num_of_bus + num_of_branch + 1 to num_of_bus + num_of_branch + num_of_gen
%
function result = CC_DCOPF_Solver(Hmpc, CI_parameter)
    % ex: case_id = modified_pglib_opf_case118_ieee___R1
    %
    %
    [num_of_bus, zz] = size(Hmpc.bus);
    [num_of_gen, zz] = size(Hmpc.gen);
    [num_of_branch, zz] = size(Hmpc.branch);
    %
    total_num_of_variables = num_of_bus + num_of_branch + num_of_gen;
    %
    Aeq_theta_to_T = zeros(num_of_branch, total_num_of_variables);
    for i = 1:num_of_branch
        curr_head = Hmpc.branch(i,1);
        curr_tail = Hmpc.branch(i,2);
        curr_x    = Hmpc.branch(i,4);
        %
        Aeq_theta_to_T(i,curr_head) = (1/curr_x);
        Aeq_theta_to_T(i,curr_tail) = (-1/curr_x);
        Aeq_theta_to_T(i,num_of_bus + i) = -1;
    end
    beq_theta_to_T = zeros(num_of_branch, 1);
    %
    % A_load_balance * variables > b_load_balance
    A_load_balance = zeros(num_of_bus, total_num_of_variables);
    for i = 1:num_of_branch
        curr_head = Hmpc.branch(i,1);
        curr_tail = Hmpc.branch(i,2);
        A_load_balance(curr_head,num_of_bus + i) = -1; % branch i in outflow from bus curr_head  
        A_load_balance(curr_tail,num_of_bus + i) = +1;  % branch i in inflow to bus curr_tail  
    end
    for i = 1:num_of_gen
        curr_node_of_gen = Hmpc.gen(i,1);
        A_load_balance(curr_node_of_gen,num_of_bus + num_of_branch + i) = 1; % generator located at this node 
    end
    vec_of_nodal_loads = Hmpc.bus(:,3);
    b_load_balance = vec_of_nodal_loads;
    %
    % A_confidence_interval * var > b_confidence_interval
    A_confidence_interval = zeros(1,total_num_of_variables);
    A_confidence_interval(num_of_bus + num_of_branch + 1:end) = ones(1,num_of_gen);
    b_confidence_interval = CI_parameter;
    %
    % lb and ub
    lb_theta = -pi*ones(num_of_bus,1);
    ub_theta = pi*ones(num_of_bus,1);
    lb_T = -1*Hmpc.branch(:,6);
    ub_T = Hmpc.branch(:,6);
    lb_Pg = Hmpc.gen(:,10);
    ub_Pg = Hmpc.gen(:,9);
    %% Writing the Quadratic OPF formulation
    % finding the constraint matrices
    A = [-1*A_load_balance; -1*A_confidence_interval];
    b = [-1*b_load_balance; -1*b_confidence_interval];
    %
    anorm=max(vecnorm(A));
    A=A./anorm;
    b=b./anorm;
    %
    Aeq = Aeq_theta_to_T;
    beq = beq_theta_to_T;
    %
    aeqnorm=max(vecnorm(Aeq));
    Aeq=Aeq./aeqnorm;
    beq=beq./aeqnorm;
    %
    lb = [lb_theta;lb_T;lb_Pg];
    ub = [ub_theta;ub_T;ub_Pg];
    %
%     ulqnorm=max([vecnorm(ub),vecnorm(lb)]);
%     lb=lb./ulqnorm;
%     ub=ub./ulqnorm;
    %
%     mpopt = optimoptions('quadprog','Algorithm','active-set');
    %
    % finding the H and f (quadratic and linear cost matrices)
    H = zeros(total_num_of_variables,total_num_of_variables);
    for i = 1:num_of_gen
        H(num_of_bus + num_of_branch + i,num_of_bus + num_of_branch + i) = Hmpc.gencost(i,5);
    end
    f = zeros(num_of_gen,1);
    for i = 1:num_of_gen
        f(num_of_bus + num_of_branch + i,1) = Hmpc.gencost(i,6);
    end
    %
    %
    [x,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    %
    result = struct;
    result.theta = x(1:num_of_bus);
    result.T = x(num_of_bus+1:num_of_bus + num_of_branch);
    result.Pg = x(num_of_bus + num_of_branch + 1:num_of_bus + num_of_branch + num_of_gen);
    result.LMP = lambda.ineqlin(1:num_of_bus);
    result.dual_CI = lambda.ineqlin(num_of_bus + 1);
    result.dual_T_u = lambda.upper(num_of_bus + 1: num_of_bus + num_of_branch);
    result.dual_Pg_u = lambda.upper(num_of_bus + num_of_branch + 1:num_of_bus + num_of_branch + num_of_gen);
    result.dual_T_l = lambda.lower(num_of_bus + 1: num_of_bus + num_of_branch);
    result.dual_Pg_l = lambda.lower(num_of_bus + num_of_branch + 1:num_of_bus + num_of_branch + num_of_gen);
    if exitflag == 1
        result.success = 1;
    else
        result.success = 0;
    end
end
