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
        stock_1.append(np.random.uniform(-0.1, .1))
        stock_2.append(np.random.uniform(-0.1, .1))
        stock_3.append(np.random.uniform(-0.1, .1))
<<<<<<< HEAD
        dummy.append(0.0)
=======
        dummy.append(0.05)


>>>>>>> aeb19fa44cb203acc7445fc928762e797cb41e13
    #make into pandas obj
    data = pd.DataFrame(
        {'stock_1': stock_1,
         'stock_2': stock_2,
         'stock_3': stock_3,
         'dummy': dummy
        })
    return data

#create meta data that is markov
def create_markov(days):
    #stock 1 - low reward more predicable
    stock_1_rates = np.array([-0.05, 0.0, 0.05])
    stock_1_transition_matrix = np.array([[0.9, 0.05, 0.05],
                                          [0.05, 0.9, 0.05],
                                          [0.05, 0.05, 0.9]])
    # stock 2 - larger reward less predictable
    stock_2_rates = np.array([-0.1, 0.0, 0.1])
    stock_2_transition_matrix = np.array([[0.8, 0.1, 0.1],
                                          [0.1, 0.8, 0.1],
                                          [0.1, 0.1, 0.8]])
    # stock 3 - big reward but more random
    stock_3_rates = np.array([-0.25, 0.0, 0.25])
    stock_3_transition_matrix = np.array([[0.2, 0.4, 0.4],
                                          [0.4, 0.2, 0.4],
                                          [0.4, 0.4, 0.2]])
<<<<<<< HEAD
    dummy = []
    stock_1 = []
    stock_2 = []
    stock_3 = []
=======

    # stock 4 - big reward and more predicable
    stock_4_rates = np.array([-0.25, 0.0, 0.25])
    stock_4_transition_matrix = np.array([[0.9, 0.05, 0.05],
                                          [0.05, 0.9, 0.05],
                                          [0.05, 0.05, 0.9]])

    #dummy = []

    stock_1 = []
    stock_2 = []
    stock_3 = []
    stock_4 = []

>>>>>>> aeb19fa44cb203acc7445fc928762e797cb41e13
    #init previous value for markov chains
    index_1 = 0
    index_2 = 0
    index_3 = 0
    index_4 = 0
    #randomly choose for each day
    for _ in range(days):
        stock_1.append(np.random.choice(a=stock_1_rates, p=stock_1_transition_matrix[index_1]))
        index_1 = np.where(stock_1_rates == stock_1[-1])[0][0]
        stock_2.append(np.random.choice(a=stock_2_rates, p=stock_2_transition_matrix[index_2]))
        index_2 = np.where(stock_2_rates == stock_2[-1])[0][0]
        #stock_2_transition_matrix = np.dot(stock_2_transition_matrix,stock_2_transition_matrix)
        stock_3.append(np.random.choice(a=stock_3_rates, p=stock_3_transition_matrix[index_3]))
        index_3 = np.where(stock_3_rates == stock_3[-1])[0][0]
        #stock_3_transition_matrix = np.dot(stock_3_transition_matrix,stock_3_transition_matrix)
<<<<<<< HEAD
        dummy.append(0.0)
=======

        stock_4.append(np.random.choice(a=stock_4_rates, p=stock_4_transition_matrix[index_4]))
        index_4 = np.where(stock_4_rates == stock_4[-1])[0][0]

        #dummy.append(0.0)


>>>>>>> aeb19fa44cb203acc7445fc928762e797cb41e13
    #make into pandas obj
    data = pd.DataFrame(
        {'stock_1': stock_1,
         'stock_2': stock_2,
         'stock_3': stock_3,
         'stock_4': stock_4,
         #'dummy': dummy
        })
    return data

def create_markov_memory_2(days):
    #stock 1 values and 1st order and 2nd order transition matrices
    stock_1_rates = np.array([-0.05, 0.0, 0.05])
    stock_1_single_step = np.array([[0.9, 0.05, 0.05],
                                    [0.05, 0.9, 0.05],
                                    [0.05, 0.05, 0.9]])
    stock_1_double_step = np.dot(stock_1_single_step, stock_1_single_step)
    #stock 2 values and 1st order and 2nd order transition matrices
    stock_2_rates = np.array([-0.1, 0.0, 0.1])
    stock_2_single_step = np.array([[0.8, 0.1, 0.1],
                                    [0.1, 0.8, 0.1],
                                    [0.1, 0.1, 0.8]])
    stock_2_double_step = np.dot(stock_2_single_step, stock_2_single_step)
    print(stock_2_double_step)


    #stock_3_rates = np.array([-0.25, 0.0, 0.25])

    return 

if __name__ == '__main__':
    #example on how you can call it
    #return_rates = get_data()
    #print('Data Shape: ', return_rates.shape)
    #print('Column Headers are: ', list(return_rates))
    #print('First 10 return rates for IBM: \n', return_rates['ibm'].iloc[0:11])
    #return_rates_train, return_rates_test = split_data(return_rates)
    #print('Training Shape: ', return_rates_train.shape, '\nTesting Shape: ', return_rates_test.shape)

    #create_iid(15)
    markov_data = create_markov_memory_2(50)
