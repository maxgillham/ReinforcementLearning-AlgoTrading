import pandas as pd
import numpy as np
import os

#main method to get the return rates for the three given stocks over time
def get_data():
    os.chdir('..')
    ibm = compute_return_rates(sort_by_recent(load_csv('data/daily_IBM.csv')))
    msft = compute_return_rates(sort_by_recent(load_csv('data/daily_MSFT.csv')))
    qcom = compute_return_rates(sort_by_recent(load_csv('data/daily_QCOM.csv')))
    data = pd.concat([ibm, msft, qcom], axis=1, keys=['ibm', 'msft', 'qcom'])
    data['dummy'] = [0.0]*len(ibm)
    os.chdir('./Q-learning')
    return data


#load a csv into a pandas data frame
#columns are timestamp, open, high, low, close, volume
def load_csv(path):
    return pd.read_csv(path)

#sort data by accending dates
def sort_by_recent(df):
    return df.sort_values('timestamp')


#divide data into training and testing data
def split_data(df):
    train_data = df.iloc[0:df.shape[0]-100]
    test_data = df.iloc[df.shape[0]-100:]
    return train_data, test_data

#convert to rate of return
def compute_return_rates(df):
    return ((df['close'] - df['open']) / df['open'])*100

#rounds rate of return to 4 decimal palces to discritize values
def round_return_rate(df):
    return (df.round(2))/100

#will return max and min for when defining space ranges
def get_max_and_min(df):
    return df.values.max(), df.values.min()
#makes numers like 5, 10, 15.... when given 5
def round_to_base(value, base):
    return int(base * round(value/base))

#create meta data that is i.i.d
def create_iid(days):

    #init stock lists
    stock_1 = []
    stock_2 = []
    stock_3 = []
    dummy = []

    #randomly choose for each day
    for _ in range(days):
        stock_1.append(np.random.choice([-.1, 0, .1]))
        stock_2.append(np.random.choice([-0.25, 0, .25]))
        stock_3.append(np.random.choice([-0.05, 0, 0.05]))
        dummy.append(0.0)


    #make into pandas obj
    data = pd.DataFrame(
        {'stock_1': stock_1,
         'stock_2': stock_2,
         'stock_3': stock_3,
         'dummy': dummy
        })

    return data

if __name__ == '__main__':
    #example on how you can call it
    #return_rates = get_data()
    #print('Data Shape: ', return_rates.shape)
    #print('Column Headers are: ', list(return_rates))
    #print('First 10 return rates for IBM: \n', return_rates['ibm'].iloc[0:11])
    #return_rates_train, return_rates_test = split_data(return_rates)
    #print('Training Shape: ', return_rates_train.shape, '\nTesting Shape: ', return_rates_test.shape)

    create_iid(15)
