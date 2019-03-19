import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans

# main method to get the return rates for the three given stocks over time
def get_data():
    ibm = compute_return_rates(sort_by_recent(load_csv('../data/daily_IBM.csv')))
    msft = compute_return_rates(sort_by_recent(load_csv('../data/daily_MSFT.csv')))
    qcom = compute_return_rates(sort_by_recent(load_csv('../data/daily_QCOM.csv')))
    data = pd.concat([msft], axis=1, keys=['msft'])
    data['dummy'] = [0.0]*len(ibm)
    return data


# load a csv into a pandas data frame
# columns are timestamp, open, high, low, close, volume
def load_csv(path):
    return pd.read_csv(path)

# sort data by accending dates
def sort_by_recent(df):
    return df.sort_values('timestamp')


# divide data into training and testing data
def split_data(df):
    train_data = df.iloc[0:df.shape[0]-1000]
    test_data = df.iloc[df.shape[0]-1000:]
    return train_data, test_data

# convert to rate of return
def compute_return_rates(df):
    return ((df['close'] - df['open']) / df['open'])*100

# rounds rate of return to 4 decimal palces to discritize values
def round_return_rate(df):
    return (df.round(2))/100

# will return max and min for when defining space ranges
def get_max_and_min(df):
    return df.values.max(), df.values.min()
# makes numers like 5, 10, 15.... when given 5
def round_to_base(value, base):
    return int(base * round(value/base))

# create meta data that is i.i.d
def create_iid(days):
    #init stock lists
    stock_1 = []
    dummy = []
    #randomly choose for each day
    for _ in range(days):
        stock_1.append(np.random.choice([-.01, 0, 0.01]))
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
    stock_1_rates = np.array([-0.0222, 0.0003, 0.0248])
    stock_1_transition_matrix = np.array([[0.2511, 0.5038, 0.2451],
                                          [0.1587, 0.7099, 0.1314],
                                          [0.2410, 0.5559, 0.2031]])
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

def create_custom_markov_samples(days, return_rates, transition_matrix):
    # init stock and dummy stock representing choice of not investing
    stock = []
    dummy = [0]*days
    # init previous markov value
    index = 0
    for _ in range(days):
        stock.append(np.random.choice(a=return_rates, p=transition_matrix[index]))
        index = np.where(return_rates == stock[-1])[0][0]
    # put into pandas dataframe
    data = pd.DataFrame(
        {'stock_1': stock,
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

# 2 markov sources and 2 i.i.d sources
def create_markov_iid_mix(days):
    # stock 1 - low reward more predicable
    stock_1_rates = np.array([-0.05, 0.0, 0.05])
    stock_1_transition_matrix = np.array([[0.9, 0.05, 0.05],
                                          [0.05, 0.9, 0.05],
                                          [0.05, 0.05, 0.9]])
    # init lists for 2 stocks
    stock_1 = []
    stock_2 = []
    # init index for markov sources
    index_1 = 0
    # create instances of each source for num of days
    for _ in range(days):
        stock_1.append(np.random.choice(a=stock_1_rates, p=stock_1_transition_matrix[index_1]))
        index_1 = np.where(stock_1_rates == stock_1[-1])[0][0]
        stock_2.append(np.random.choice([-0.1,0.1], p=[.5,.5]))
    # make into pandas obj
    data = pd.DataFrame(
        {'MC': stock_1,
         'IID': stock_2
        })
    return data

# Returns optimal codebook and partition given a 1-D numpy array of values
# Using a 3 level uniform quantizer
# If passing a column on data from pandas obj, pass paramenter as df['msft'].values
def quantize(data):
    kmeans = KMeans(n_clusters=3).fit(data.reshape(-1,1))
    codebook = np.sort(kmeans.cluster_centers_, axis=0).reshape(3)
    bounds = [np.mean(codebook[:2]), np.mean(codebook[1:])]
    return codebook, bounds

# Returns a single step transition matrix given a 1D numpy array of values and
# The bounds for quantizing the values into 3 states
def empirical_transition_matrix(data, bounds):
    # get initial state
    if data[0] < bounds[0]:
        state = 0
    elif data[0] > bounds[1]:
        state = 2
    else: state = 1
    # init transition matrix
    P = np.zeros((3,3))
    # iterate though data in chronological order and increment number
    # of transitions
    for i in range(1, data.shape[0]):
        if data[i] < bounds[0]:
            P[state, 0] += 1
            state = 0
        elif data[i] > bounds[1]:
            P[state, 2] += 1
            state = 2
        else:
            P[state, 1] += 1
            state = 1
    # Divide counts for each state by sum of row to
    # Turn into a stochastic matrix
    P[0] /= np.sum(P, axis=1)[0]
    P[1] /= np.sum(P, axis=1)[1]
    P[2] /= np.sum(P, axis=1)[2]
    return P
