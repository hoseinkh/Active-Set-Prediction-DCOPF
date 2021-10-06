%
% Programmer: Hossein Khazaei (husein.khazaei@gmail.com)
% Date: May 09, 2020
%
% This M-file is as a guideline on how to use MATPOWER to perform DC-OPF
%
%
%
%% Define the model
% Create an empty struct as a "mpc" file to save the network in!
My_Network = struct;
%
% define base MVA
My_Network.baseMVA = 100;
%
%
% Those entries with "-" are not required when creating a network, and ...
% ... they will be filled automatically when solving OPF!
%
% % Adding the data of the buses: (note that fixed demand is defined on column 3 of each node!)  
% column index:      1       2       3    4   5   6     7       8   9     10     11    12    13     14     15     16        17      
% column names:    BUS_I, BUS_TYPE, PD,  QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN    
My_Network.bus = [  %1       3       0    0   0   0     1       1   0     10      1   1.2   0.8      -      -      -         -   ; this is an example bus 
                     1       3       0    0   0   0     1       1   0     10      1   1.2   0.8;
                     2       2       0    0   0   0     1       1   0     10      1   1.2   0.8;
                     3       1      150   0   0   0     1       1   0     10      1   1.2   0.8];
%
%
%
% % Adding the branch data
% column index:          1      2     3      4     5        6            7           8      9    10      11        12      13     14  15  16  17   18      19       20         21   
% column names:       F_BUS, T_BUS, BR_R,  BR_X, BR_B,    RATE_A,     RATE_B,     RATE_C,  TAP, SHIFT, BT_STATUS, ANGMIN, ANGMAX, PF, QF, PT, QT, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX      
My_Network.branch = [   %1      2     0     0.1    0       70            70         70      0     0       1        -360    360     -   -   -   -    -       -        -          -;  this is an example line
                         1      2     0     0.1    0       inf          inf        inf      0     0       1        -360    360;
                         1      3     0     0.3    0        40           40         40      0     0       1        -360    360;
                         2      3     0     0.1    0       inf          inf        inf      0     0       1        -360    360];
%
%
% % Adding the data of the generators:
% column index:      1       2      3      4       5      6            7                 8          9      10      11    12    13      14      15      16       17       18       19       20     21    22       23       24       25            
% column names:    GEN_BUS, PG,    QG,   QMAX,   QMIN,   VG,         MBASE,          GEN_STATUS,  PMAX,   PMIN,   PC1,  PC2, QC1MIN, QC1MAX, QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN     
My_Network.gen = [  %1      10      0     800      0      1    My_Network.baseMVA        1         800      0       0   800     0      800      0      800     1000     1000     1000     1000     0     0        0        0        0 ;  this is an example line 
                     1      10      0     100      0      1    My_Network.baseMVA        1         100      0       0   100     0      100      0      100     1000     1000     1000     1000     0     0        0        0        0    ; 
                     2      10      0     200      0      1    My_Network.baseMVA        1         200      0       0   200     0      200      0      200     1000     1000     1000     1000     0     0        0        0        0    ]; 
%
%
%
% % Adding the data of the generators' cost functions:  (the data of each row is for the corresponding generator on the same row of My_Network.gen)  
% column index:          1       2            3         4        5      
% column names:        Model, STARTUP,    SHUTDOWN,   NCOST,   COST,      
My_Network.gencost = [   2       0            0         2       60    0; 
                         2       0            0         2      120    0];
%
%
%
rundcopf(My_Network)






