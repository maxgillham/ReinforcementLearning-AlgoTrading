import pandas as pd
import numpy as np
import os
import itertools
from sklearn.cluster import KMeans


def get_data():
    """
    This method loads historical candles data from csv files, sorts by time stamp,
    computes daily return rate and concatenates into a pandas.DataFrame for
    IBM, MSFT, QCOM and the choice of not investing (constant return rate of 0).

    Returns
    -------
    data: pandas.DataFrame

    """
    ibm = compute_return_rates(sort_by_recent(load_csv('../data/daily_IBM.csv')))
    msft = compute_return_rates(sort_by_recent(load_csv('../data/daily_MSFT.csv')))
    qcom = compute_return_rates(sort_by_recent(load_csv('../data/daily_QCOM.csv')))
    data = pd.concat([ibm, msft], axis=1, keys=['ibm', 'msft'])
    data['dummy'] = [0.0]*len(ibm)
    return data

def split_data(df):
    """
    Splits a source of data into training and testing segments.

    Parameters
    ----------
    df: pandas.DataFrame
        The entire set of historical return rates

    Returns
    -------
    train_data, test_data: pandas.DataFrame, pandas.DataFrame

    """
    train_data = df.iloc[0:df.shape[0]-1000]
    test_data = df.iloc[df.shape[0]-1000:]
    return train_data, test_data

# convert to rate of return
def compute_return_rates(df):
    """
    Computes to ROI for 1 day.

    Parameters
    ----------
    df: pandas.DataFrame
        The candles data from the csv files.

    Returns
    -------
    data: pandas.DataFrame

    """
    return ((df['close'] - df['open']) / df['open'])*100

def create_iid(days):
    """
    Creates a sequence of values to represent daily return rates according to a
    IID process and forms a DataFrame with a constant return rate of 0 to
    represent the choice of not investing.

    Parameters
    ----------
    days: int
        The number of days you wish to generate IID data for.

    Returns
    -------
    data: pandas.DataFrame

    """
    # Init stock lists
    stock_1 = []
    dummy = []
    # Randomly choose for each day
    for _ in range(days):
        stock_1.append(np.random.choice([-.01, 0, 0.01]))
        dummy.append(0.0)
    # Make into pandas obj
    data = pd.DataFrame(
        {'stock_1': stock_1,
         'dummy': dummy
        })
    return data

def create_markov(days):
    """
    Creates a sequence of values to represent daily return rates according to a
    Markov Memory 1 distribution and forms a DataFrame with the choice of not investing.

    Parameters
    ----------
    days: int
        The number of days you wish to generete markovian data for.

    Returns
    -------
    data: pandas.DataFrame

    """
    stock_1_rates = np.array([-0.0222, 0.0003, 0.0248])
    stock_1_transition_matrix = np.array([[0.2511, 0.5038, 0.2451],
                                          [0.1587, 0.7099, 0.1314],
                                          [0.2410, 0.5559, 0.2031]])
    # Init stock 1 values and dummy value
    stock_1 = []
    dummy = []

    # Init previous value for markov chains
    index_1 = 0
    # Randomly choose for each day
    for _ in range(days):
        stock_1.append(np.random.choice(a=stock_1_rates, p=stock_1_transition_matrix[index_1]))
        index_1 = np.where(stock_1_rates == stock_1[-1])[0][0]
        dummy.append(0)

    # Make into pandas obj
    data = pd.DataFrame(
        {'stock_1': stock_1,
         'dummy': dummy
        })
    return data

def create_custom_markov_samples(days, return_rates, transition_matrix):
    """
    Creates a sequence of values to represent daily return rates acording to a
    specified transition matrix and set of possible return rates.

    Parameters
    ----------
    days: int
        The number of days you wish to generate data for.

    return_rates: np.array(1,3)
        The set of possible return rates.

    transition_matrix: np.array(3,3)
        The transition matrix for return rate values.

    Returns
    -------
    data: pandas.DataFrame

    """
    # Init stock and dummy stock representing choice of not investing
    stock = []
    dummy = [0]*days
    # Init previous markov value
    index = 0
    for _ in range(days):
        stock.append(np.random.choice(a=return_rates, p=transition_matrix[index]))
        index = np.where(return_rates == stock[-1])[0][0]
    # Put into pandas dataframe
    data = pd.DataFrame(
        {'stock_1': stock,
         'dummy': dummy
        })
    return data

def create_markov_memory_2(days):
    """
    Creates a sequence of values to resemble return rates according to a Markov
    Memory 2 distribution.

    Parameters
    ----------
    days: int
        The number of days you wish to generate data for.

    Returns
    -------
    data: pandas.DataFrame

    """
    stock_1_rates = np.array([-.1, 0, 0.1]) # Possible return rates
    # Memory two transistion matrix, shape 3x3
    stock_1_transition_matrix = np.array([[[0.2, 0.3, 0.5],
                                           [0.1, 0.2, 0.7],
                                           [0.4, 0.2, 0.4]],
                                          [[0.2, 0.1, 0.7],
                                           [0.1, 0.1, 0.8],
                                           [0.5, 0.2, 0.3]],
                                          [[0.4, 0.6, 0.0],
                                           [0.2, 0.5, 0.3],
                                           [0.0, 0.5, 0.5]]])
    # Using a list to permit item assignment
    stock_1_prev_indices = [0,0]
    stock_1 = []
    dummy = []
    for _ in range(days):
        stock_1.append(np.random.choice(a=stock_1_rates, p=stock_1_transition_matrix[stock_1_prev_indices[0], stock_1_prev_indices[1]]))
        # Assign 2 most recent value to most recent value
        stock_1_prev_indices[0] = stock_1_prev_indices[1]
        # Assign lastest value to most recent
        stock_1_prev_indices[1] = np.where(stock_1_rates == stock_1[-1])[0][0]
        dummy.append(0.0)
    # Make into pandas obj
    data = pd.DataFrame(
        {'stock_1': stock_1,
         'dummy': dummy
        })
    return data

def create_markov_iid_mix(days):
    """
    Creates a sequence of values to reing to a IID process and Markov process.

    Parameters
    ----------
    days: int
        The number of days to generate data for.

    Returns
    -------
    data: pandas.DataFrame

    """
    # Stock 1 - low reward more predicable
    stock_1_rates = np.array([-0.05, 0.0, 0.05])
    stock_1_transition_matrix = np.array([[0.9, 0.05, 0.05],
                                          [0.05, 0.9, 0.05],
                                          [0.05, 0.05, 0.9]])
    # Init lists for 2 stocks
    stock_1 = []
    stock_2 = []
    # Init index for markov sources
    index_1 = 0
    # Create instances of each source for num of days
    for _ in range(days):
        stock_1.append(np.random.choice(a=stock_1_rates, p=stock_1_transition_matrix[index_1]))
        index_1 = np.where(stock_1_rates == stock_1[-1])[0][0]
        stock_2.append(np.random.choice([-0.1,0.1], p=[.5,.5]))
    # Make into pandas obj
    data = pd.DataFrame(
        {'MC': stock_1,
         'IID': stock_2
        })
    return data

def quantize(data):
    """
    Computes the optimal codebook and partition for a 3 level qantizer given
    a training set.

    Parameters
    ----------
    data: np.array(1, n) where n is the number of samples
        The training set to compute a quantizer for.

    Returns
    -------
    codebook, bounds: np.array(1,3), list[float, float]

    """
    kmeans = KMeans(n_clusters=3).fit(data.reshape(-1,1))
    codebook = np.sort(kmeans.cluster_centers_, axis=0).reshape(3)
    bounds = [np.mean(codebook[:2]), np.mean(codebook[1:])]
    return codebook, bounds


def empirical_transition_matrix(data, bounds):
    """
    Computes an empirical transition matrix over training data.

    Parameters
    ----------
    data: np.array(1,)
        The set of return rates.

    bounds: list[float, float]
        The 3 level quantizer bounds.

    Returns
    -------
    P: np.array(3,3)

    """
    # Get initial state
    if data[0] < bounds[0]:
        state = 0
    elif data[0] > bounds[1]:
        state = 2
    else: state = 1
    # Init transition matrix
    P = np.zeros((3,3))
    # Iterate though data in chronological order and increment number
    # Of transitions
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


def define_actions(n_stocks, options):
    """
    Defines the set of feasible control actions under the condition of the
    sum of capital allocations equaling 1.

    Parameters
    ----------
    n_stocks: int
        The number of stocks in the portfolio.

    options: list[floats]
        The set of possible partition values.

    Returns
    -------
    actions: list[(float, float, float)]

    """
    actions = []
    for i in itertools.product(options, repeat=n_stocks):
        if sum(i)==1: actions.append(i)
    return actions

def define_observations(n_stocks, options):
    """
    Defines the observation space for the Q learning agent.

    Parameters
    ----------
    n_stocks: int
        The number of stocks in the portfolio.

    options: list
        The number of observations for 1 single stock

    Returns
    -------
    obs: list[]

    """
    obs = []
    for i in itertools.product(options, repeat=n_stocks):
        i = i + (0,)
        obs.append(list(i))
    return obs


# Rounds return rates
def round_return_rate(df):
    return (df.round(2))/100

# Will return max and min for when defining space ranges
def get_max_and_min(df):
    return df.values.max(), df.values.min()

# Makes numers like 5, 10, 15.... when given 5
def round_to_base(value, base):
    return int(base * round(value/base))

# Load a csv into a pandas data frame
# Columns are timestamp, open, high, low, close, volume
def load_csv(path):
    return pd.read_csv(path)

# Sort data by accending dates
def sort_by_recent(df):
    return df.sort_values('timestamp')
