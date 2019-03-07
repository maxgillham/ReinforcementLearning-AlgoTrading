%% MTHE 493
% Determining a policy for a Markov memory 1 source 
% By: Bryony

%% Markov Memory 1 with a generated policy
P = [ 0.8 0.1 0.1; %probability transition matrix
      0.1 0.8 0.1;
      0.1 0.1 0.8];
states = [-1 0 1];  
n = 3; %number of stocks and bonds
P1 = P(1,:);
P2 = P(2,:);
P3 = P(3,:);
cvx_begin
    variable u_T1(1,n+1)
    functn = 0;
    steps = 10;
    lowerReturn = 0;
    upperReturn = 1;
    interval = (upperReturn - lowerReturn) / steps;
    r_T = zeros(1,n+1); %the bank then the uniform stocks
    for i = lowerReturn:interval:upperReturn
        r_T(2:n+1) = i;
        if i < -0.001
          prob1 = P1(1); 
        elseif i < 0.001
          prob1 = P1(2);  
        else
          prob1 = P1(3);  
        end
        functn = functn + prob1*log(u_T1*(1+r_T)');
    end
    
    maximize(functn)
    subject to
        sum(u_T1) == 1
        u_T1 >= 0 
        u_T1 <= 1
cvx_end
cvx_begin
    variable u_T2(1,n+1)
    functn = 0;
    steps = 10;
    lowerReturn = 0;
    upperReturn = 1;
    interval = (upperReturn - lowerReturn) / steps;
    r_T = zeros(1,n+1); %the bank then the uniform stocks
    for i = lowerReturn:interval:upperReturn
        r_T(2:n+1) = i;
        if i < -0.001
          prob2 = P2(1); 
        elseif i < 0.001
          prob2 = P2(2);  
        else
          prob2 = P2(3);  
        end
        functn = functn + prob2*log(u_T2*(1+r_T)');
    end
    
    maximize(functn)
    subject to
        sum(u_T2) == 1
        u_T2 >= 0 
        u_T2 <= 1
cvx_end
cvx_begin
    variable u_T3(1,n+1)
    functn = 0;
    steps = 10;
    lowerReturn = 0;
    upperReturn = 1;
    interval = (upperReturn - lowerReturn) / steps;
    r_T = zeros(1,n+1); %the bank then the uniform stocks
    for i = lowerReturn:interval:upperReturn
        r_T(2:n+1) = i;
        if i < -0.001
          prob3 = P3(1); 
        elseif i < 0.001
          prob3 = P3(2);  
        else
          prob3 = P3(3);  
        end
        functn = functn + prob3*log(u_T3*(1+r_T)');
    end
    
    maximize(functn)
    subject to
        sum(u_T3) == 1
        u_T3 >= 0 
        u_T3 <= 1
cvx_end