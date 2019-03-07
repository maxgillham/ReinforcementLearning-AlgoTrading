%% Mathematical Analysis of the Proportion Distribution
% MTHE 493: 2018-2019
% Bryony and Eric
clc
clear

% Variables:
% - X_T = capital at time t
% - u_T^i = proportion for stock i at time T
% - r_T^i = return rate at time t for stock i
% - c_t1 = capital at time t+1
% - n = number of stocks
% Objective:
% - want to maximize the expected value of the log of the cost function
% - can then just say max E[log(X_T)] = max E[log(sum_0^n
% (u_T^i*(1+r_T^i)))] since the rest are constant
%   - subject to:
%      - {u_T^i} is in [0,1]
%      - sum_0^n (u_T^i) = 1
% - alternatively, minimize the negative of that function


%% IID with 1 stock
% n = 1; %number of stocks and bonds
% 
% cvx_begin
%     variable u_T(1,n+1)
%     functn = 0;
%     steps = 10;
%     prob = 1/steps;
%     lowerReturn = -0.5;
%     upperReturn = 0.5;
%     interval = (upperReturn - lowerReturn) / steps;
%     r_T = zeros(1,n+1); %the bank then the uniform stocks
%     for i = lowerReturn:interval:upperReturn
%         r_T(2:n+1) = i; 
%         functn = functn + prob*log(u_T*(1+r_T)');
%     end
%     
%     maximize(functn)
%     subject to
%         sum(u_T) == 1
%         u_T >= 0 
%         u_T <= 1
% cvx_end
% 
% %% IID with 2 stocks
% n = 2; %number of stocks and bonds
% 
% cvx_begin
%     variable u_T(1,n+1)
%     functn = 0;
%     steps = 10;
%     prob = 1/steps;
%     lowerReturn = 0;
%     upperReturn = 1;
%     interval = (upperReturn - lowerReturn) / steps;
%     r_T = zeros(1,n+1); %the bank then the uniform stocks
%     for i = lowerReturn:interval:upperReturn
%         for j = lowerReturn:interval:upperReturn
%             r_T(2:n+1) = [i j]; 
%             functn = functn + prob*prob*log(u_T*(1+r_T)');
%         end
%     end
%     
%     maximize(functn)
%     subject to
%         sum(u_T) == 1
%         u_T >= 0 
%         u_T <= 1
% cvx_end
% 
% %% Markov Memory 1 with one stock
P = [ 0.1 0.8 0.1]; %probability transition matrix
states = [-0.3 0 0.4];  
cvx_begin
    variable u_T(1,2)
    functn = 0;
    r_T = zeros(1,2); %the bank then the one stocks
    for i = 1:3
        r_T(2) = states(i)
        functn = functn + P(i)*log(u_T*(1+r_T)');
    end
    
    maximize(functn)
    subject to
        sum(u_T) == 1
        u_T >= 0 
        u_T <= 1
cvx_end
% 
% %% Markov Memory 1 with two stocks
% P = [ 0.8 0.1 0.1; %probability transition matrix for stock 1
%       0.1 0.8 0.1;
%       0.1 0.1 0.8];
% states1 = [-1 0 1]; %associated state space for stock 1
% P2 = [ 0.4 0.3 0.3; %probability transition matrix for stock 2
%       0.3 0.4 0.3;
%       0.3 0.3 0.4];
% states2 = [-1 0 1]; %associated state space for stock 2
% cvx_begin
%     variable u_T(1,3)
%     functn = 0;
%     r_T = zeros(1,3); %the bank then the two stocks
%     for i = 1:size(P,1)
%         for j = 1:size(P,2)
%             for k = 1:size(P2,1)
%                 for l = 1:size(P2,2)
%                     r_T(2:3) = [states1(j) states2(l)];
%                     functn = functn + P(i,j)*P2(k,l)*log(u_T*(1+r_T)');
%                 end
%             end
%         end
%     end
%     
%     maximize(functn)
%     subject to
%         sum(u_T) == 1
%         u_T >= 0 
%         u_T <= 1
% cvx_end
% %% Markov Memory 2
% %working with one stock
% %probability transition matrix
% P = zeros(3);
% P(:,:,1) = [ 0.8 0.1 0.1; 
%          0.1 0.8 0.1;
%          0.1 0.1 0.8];
% P(:,:,2) = [ 0.6 0.2 0.2; 
%          0.2 0.6 0.2;
%          0.2 0.2 0.6];
% P(:,:,3) = [ 0.4 0.3 0.3; 
%          0.3 0.4 0.3;
%          0.3 0.3 0.4];
% states = [-0.25 0 1]; %associated state space for stock 1
% 
% cvx_begin
%     variable u_T(1,2)
%     functn = 0;
%     r_T = zeros(1,2); %the bank then the two stocks
%     for i = 1:size(P,1)
%         for j = 1:size(P,2)
%             for k = 1:size(P,3)
%                 r_T(2) = states(k);
%                 functn = functn + P(i,j,k)*log(u_T*(1+r_T)');
%             end
%         end
%     end
%     
%     maximize(functn)
%     subject to
%         sum(u_T) == 1
%         u_T >= 0 
%         u_T <= 1
% cvx_end