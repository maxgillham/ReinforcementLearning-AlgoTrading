function [RR] = toDailyReturnRate(data)
% turns array of closing prices to a list of daily return rates
 
for i = 1:height(data)-1000
    RR(i) = ((data{i, 5} - data{i, 2})/data{i, 2});
end

end

