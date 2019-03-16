%% MTHE 493: Model Fitting
% Determining the best model for the 3 stocks: determine IID, MC1, MC2 ..
% By: Bryony Schonewille
% Date: March 3, 2019

%% IID
% clear 
% clc
% 
% nStocks = 3;
% nStates = 3; 
% binCount = zeros(nStocks,nStates);
% bounds = [-1, -0.0109, 0.0126,1];
% 
% P_IBM = zeros(1,nStates); 
% P_MSFT = zeros(1,nStates);
% P_QCOM = zeros(1,nStates);
% 
% IBM_data = sortrows(readtable('../data/daily_IBM.csv'));
% MSFT_data = sortrows(readtable('../data/daily_MSFT.csv'));
% QCOM_data = sortrows(readtable('../data/daily_QCOM.csv'));
% IBM_data = toDailyReturnRate(IBM_data); %turn to return rates
% MSFT_data = toDailyReturnRate(MSFT_data);
% QCOM_data = toDailyReturnRate(QCOM_data);
% length = length(IBM_data); %assumes all data has the same length
% 
% for i = 1:length
%     currentState = [IBM_data(i), MSFT_data(i), QCOM_data(i)];
%     for j = 1:nStocks
%         if currentState(j) < bounds(2)
%             binCount(j,1) = binCount(j,1) + 1;
%         elseif currentState(j) < bounds(3)
%             binCount(j,2) = binCount(j,2) + 1;
%         else
%             binCount(j,3) = binCount(j,3) + 1;
%         end
%     end
% end
% 
% P_IBM = binCount(1,:)/length;
% P_MSFT = binCount(2,:)/length;
% P_QCOM = binCount(3,:)/length;

%% MC1
%clear 
%clc

nStocks = 1;
nStates = 3; 
binCount = zeros(nStates,nStates);

lowerBound = -1;
upperBound = 1;
bounds = [-1, -0.0139, 0.0172,1];

P = zeros(nStates,nStates); 

Stock_data = readtable('../data/daily_APPL.csv');
Stock_data = toDailyReturnRate(Stock_data); %turn to return rates

last = length(Stock_data);

amounts = zeros(nStates);
for i = 2:last
    oldState = Stock_data(i-1);
    currentState = Stock_data(i);
    if oldState < bounds(2)
       amounts(1) = amounts(1) + 1;
       if currentState < bounds(2)
           binCount(1,1) = binCount(1,1) + 1;
       elseif currentState < bounds(3)
           binCount(1,2) = binCount(1,2) + 1;
       else
           binCount(1,3) = binCount(1,3) + 1;
       end
          
    elseif oldState > bounds(3)
           amounts(2) = amounts(2) + 1;
            if currentState < bounds(2)
                binCount(2,1) = binCount(2,1) + 1;
            elseif currentState < bounds(3)
                binCount(2,2) = binCount(2,2) + 1;
            else
                binCount(2,3) = binCount(2,3) + 1;
            end
     else
            amounts(3) = amounts(3) + 1;
            if currentState < bounds(2)
                binCount(3,1) = binCount(3,1) + 1;
            elseif currentState < bounds(3)
                binCount(3,2) = binCount(3,2) + 1;
            else
                binCount(3,3) = binCount(3,3) + 1;
            end
        end
end

for i = 1:nStates
    if amounts(1,i) ~= 0
        P(i,:) = binCount(i,:)/amounts(1,i);
    else
        P(i,:) = zeros(1,nStates);
    end
end

