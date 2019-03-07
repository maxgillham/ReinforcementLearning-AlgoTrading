%% MTHE 493: Model Fitting
% Determining the best model for the 3 stocks: determine IID, MC1, MC2 ..
% By: Bryony Schonewille
% Date: March 3, 2019

%% IID
clear 
clc

nStocks = 3;
nStates = 3; 
binCount = zeros(nStocks,nStates);
bounds = [-1, -0.001, 0.001,1];

P_IBM = zeros(1,nStates); 
P_MSFT = zeros(1,nStates);
P_QCOM = zeros(1,nStates);

IBM_data = sortrows(readtable('../data/daily_IBM.csv'));
MSFT_data = sortrows(readtable('../data/daily_MSFT.csv'));
QCOM_data = sortrows(readtable('../data/daily_QCOM.csv'));
IBM_data = toDailyReturnRate(IBM_data); %turn to return rates
MSFT_data = toDailyReturnRate(MSFT_data);
QCOM_data = toDailyReturnRate(QCOM_data);
length = length(IBM_data); %assumes all data has the same length

for i = 1:length
    currentState = [IBM_data(i), MSFT_data(i), QCOM_data(i)];
    for j = 1:nStocks
        if currentState(j) < bounds(2)
            binCount(j,1) = binCount(j,1) + 1;
        elseif currentState(j) < bounds(3)
            binCount(j,2) = binCount(j,2) + 1;
        else
            binCount(j,3) = binCount(j,3) + 1;
        end
    end
end

P_IBM = binCount(1,:)/length;
P_MSFT = binCount(2,:)/length;
P_QCOM = binCount(3,:)/length;

%% MC1
%clear 
%clc

nStocks = 3;
nStates = 3; 
binCount = zeros(nStocks,nStates,nStates);

lowerBound = -1;
upperBound = 1;
bounds = [-1, -0.1, 0.1,1];

P_IBM = zeros(nStates,nStates); 
P_MSFT = zeros(nStates,nStates);
P_QCOM = zeros(nStates,nStates);

IBM_data = sortrows(readtable('../data/daily_IBM.csv'));
MSFT_data = sortrows(readtable('../data/daily_MSFT.csv'));
QCOM_data = sortrows(readtable('../data/daily_QCOM.csv'));
IBM_data = toDailyReturnRate(IBM_data); %turn to return rates
MSFT_data = toDailyReturnRate(MSFT_data);
QCOM_data = toDailyReturnRate(QCOM_data);
%length = length(IBM_data); %assumes all data has the same length

amounts = zeros(nStocks,nStates);
for i = 2:length
    oldState = [IBM_data(i-1), MSFT_data(i-1), QCOM_data(i-1)];
    currentState = [IBM_data(i), MSFT_data(i), QCOM_data(i)];
    for j = 1:nStocks
        if oldState(j) < bounds(2)
            amounts(j,1) = amounts(j,1) + 1;
            if currentState(j) < bounds(2)
                binCount(j,1,1) = binCount(j,1,1) + 1;
            elseif currentState(j) < bounds(3)
                binCount(j,1,2) = binCount(j,1,2) + 1;
            else
                binCount(j,1,3) = binCount(j,1,3) + 1;
            end
        elseif oldState(j) < bounds(3)
            amounts(j,2) = amounts(j,2) + 1;
            if currentState(j) < bounds(2)
                binCount(j,2,1) = binCount(j,2,1) + 1;
            elseif currentState(j) < bounds(3)
                binCount(j,2,2) = binCount(j,2,2) + 1;
            else
                binCount(j,2,3) = binCount(j,2,3) + 1;
            end
        else
            amounts(j,3) = amounts(j,3) + 1;
            if currentState(j) < bounds(2)
                binCount(j,3,1) = binCount(j,3,1) + 1;
            elseif currentState(j) < bounds(3)
                binCount(j,3,2) = binCount(j,3,2) + 1;
            else
                binCount(j,3,3) = binCount(j,3,3) + 1;
            end
        end
    end
end

for i = 1:nStates
    if amounts(1,i) ~= 0
        P_IBM(i,:) = binCount(1,i,:)/amounts(1,i);
    else
        P_IBM(i,:) = zeros(1,nStates);
    end
    if amounts(2,i) ~= 0
        P_MSFT(i,:) = binCount(2,i,:)/amounts(2,i);
    else
        P_MSFT(i,:) = zeros(1,nStates);
    end
    if amounts(3,i) ~= 0
        P_QCOM(i,:) = binCount(3,i,:)/amounts(3,i);
    else
        P_QCOM(i,:) = zeros(1,nStates);
    end
end

