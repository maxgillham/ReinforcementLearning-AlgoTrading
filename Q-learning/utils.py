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
    dummy = []
    #randomly choose for each day
    for _ in range(days):
        stock_1.append(np.random.choice([-1, -.8, -.6, -.4, -.2, 0, 0.2, 0.4, 0.6, 0.8, 1]))
        dummy.append(0.0)
    #make into pandas obj
    data = pd.DataFrame(
        {'stock_1': stock_1,
         'dummy': dummy
        })
    return data

#create meta data that is markov
def create_markov(days):
    #stock 1 - markov mem 1 values and transistion matrix
    stock_1_rates = np.array([-0.2, 0, 0.2])
    stock_1_transition_matrix = np.array([[0.8, 0.1, 0.1],
                                          [0.1, 0.8, 0.1],
                                          [0.1, 0.1, 0.8]])
    #init stock 1 values and dummy value
    stock_1 = []
    dummy = []

    #init previous value for markov chains
    index_1 = 0
    #randomly choose for each day
    for _ in range(days):
        stock_1.append(np.random.choice(a=stock_1_rates, p=stock_1_transition_matrix[index_1]))
        index_1 = np.where(stock_1_rates == stock_1[-1])[0][0]
        dummy.append(0)

    #make into pandas obj
    data = pd.DataFrame(
        {'stock_1': stock_1,
         'dummy': dummy
        })
    return data

def create_markov_memory_2(days):
    stock_1_rates = np.array([-.1, 0, 0.1]) #possible return rates
    #memory two transistion matrix, shape 3x3
    stock_1_transition_matrix = np.array([[[0.2, 0.3, 0.5],
                                           [0.1, 0.2, 0.7],
                                           [0.4, 0.2, 0.4]],
                                          [[0.2, 0.1, 0.7],
                                           [0.1, 0.1, 0.8],
                                           [0.5, 0.2, 0.3]],
                                          [[0.4, 0.6, 0.0],
                                           [0.2, 0.5, 0.3],
                                           [0.0, 0.5, 0.5]]])
    #using a list to permit item assignment
    stock_1_prev_indices = [0,0]
    stock_1 = []
    dummy = []
    for _ in range(days):
        stock_1.append(np.random.choice(a=stock_1_rates, p=stock_1_transition_matrix[stock_1_prev_indices[0], stock_1_prev_indices[1]]))
        #assign 2 most recent value to most recent value
        stock_1_prev_indices[0] = stock_1_prev_indices[1]
        #assign lastest value to most recent
        stock_1_prev_indices[1] = np.where(stock_1_rates == stock_1[-1])[0][0]
        dummy.append(0.0)
    #make into pandas obj
    data = pd.DataFrame(
        {'stock_1': stock_1,
         'dummy': dummy
        })
    return data

#2 markov sources and 2 i.i.d sources
def create_markov_iid_mix(days):
    #stock 1 - low reward more predicable
    stock_1_rates = np.array([-0.05, 0.0, 0.05])
    stock_1_transition_matrix = np.array([[0.9, 0.05, 0.05],
                                          [0.05, 0.9, 0.05],
                                          [0.05, 0.05, 0.9]])
    #init lists for 2 stocks
    stock_1 = []
    stock_2 = []
    #init index for markov sources
    index_1 = 0
    #create instances of each source for num of days
    for _ in range(days):
        stock_1.append(np.random.choice(a=stock_1_rates, p=stock_1_transition_matrix[index_1]))
        index_1 = np.where(stock_1_rates == stock_1[-1])[0][0]
        stock_2.append(np.random.choice([-0.1,0.1], p=[.5,.5]))
    #make into pandas obj
    data = pd.DataFrame(
        {'MC': stock_1,
         'IID': stock_2
        })
    return data
