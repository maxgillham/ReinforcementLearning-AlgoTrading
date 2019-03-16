%generate 5000 random points from the standard normal distribution
IBM_data = readtable('../data/daily_APPL.csv');
IBM_data = toDailyReturnRate(IBM_data);

% samples = normrnd(0,1,[1,5000]);
% [partition, codebook] = lloyds(IBM_data, [-.7, 0, .1]) 

[partition,codebook] = lloyds(IBM_data,3)