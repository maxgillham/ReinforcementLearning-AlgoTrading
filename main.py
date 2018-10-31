from enviorment import TradingEnv
from utils import get_data


if __name__ == '__main__':
    train_data = get_data()
    env = TradingEnv(train_data, 100)
    print('Num Stocks', env.n_stocks)
    print('Num Steps', env.n_steps)
    print('Init Invest', env.init_capital)
    print('Action Space', env.action_space)
    print('Observation Space', env.observation_space)
