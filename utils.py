import pandas as pd

#main method to get the return rates for the three given stocks over time
def get_data():
    ibm = compute_return_rates(sort_by_recent(load_csv('data/daily_IBM.csv')))
    msft = compute_return_rates(sort_by_recent(load_csv('data/daily_MSFT.csv')))
    qcom = compute_return_rates(sort_by_recent(load_csv('data/daily_QCOM.csv')))
    return pd.concat([ibm, msft, qcom], axis=1, keys=['ibm', 'msft', 'qcom'])


#load a csv into a pandas data frame
#columns are timestamp, open, high, low, close, volume
def load_csv(path):
    return pd.read_csv(path)

#sort data by accending dates
def sort_by_recent(df):
    return df.sort_values('timestamp')


#divide data into training and testing data
def split_data(df):
    return df.iloc[0:df.shape[0]-100], df.iloc[df.shape[0]-100:]

#convert to rate of return
def compute_return_rates(df):
    return ((df['close'] - df['open']) / df['open'])

#rounds rate of return to 4 decimal palces to discritize values
def round_return_rate(df):
    return df.round(4)

#will return max and min for when defining space ranges
def get_max_and_min(df):
    return df.values.max(), df.values.min()

if __name__ == '__main__':
    #example on how you can call it
    return_rates = get_data()
    print('Data Shape: ', return_rates.shape)
    print('Column Headers are: ', list(return_rates))
    print('First 10 return rates for IBM: \n', return_rates['ibm'].iloc[0:11])
    return_rates_train, return_rates_test = split_data(return_rates)
    print('Training Shape: ', return_rates_train.shape, '\nTesting Shape: ', return_rates_test.shape)
