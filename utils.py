import pandas as pd

'''
Method to load a csv into a pandas data frame
Columns are timestamp, open, high, low, close, volume
'''
def load_csv(path):
    return pd.read_csv(path)

'''
Method to sort by timestamp, we want to train on old data so 
this makes last row most recent and first row oldest
'''
def sort_by_recent(df):
    return df.sort_values('timestamp')

'''
Method to divide data into training and testing data
'''
def split_data(df):
    return df.iloc[0:df.shape[0]-100], df.iloc[df.shape[0]-100:]


if __name__ == '__main__':
    ibm = load_csv('data/daily_IBM.csv')
    print('ibm shape', ibm.shape)
    ibm = sort_by_recent(ibm)
    ibm_train, ibm_test = split_data(ibm)
    

