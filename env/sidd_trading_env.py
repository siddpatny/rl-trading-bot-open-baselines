import random
import gym
import math
import configparser 
import pandas   as pd
import numpy    as np
from   gym      import spaces

config = configparser.ConfigParser()
config.read('config.ini')


MAX_REWARD          = float(config['ENV']['MaximumReward'])
Strategy          = str(config['ENV']['Strategy'])
starting_money      = float(config['ENV']['StartingMoney'])
bid_price_columns   = list(map(int,config['ENV']['ColumnsOfBidPrice'].split(',')))
ask_price_columns   = list(map(int,config['ENV']['ColumnsOfAskPrice'].split(',')))
current_money       = float(config['ENV']['StartingMoney'])
actions             = ['buy','sell','hold']
askPriceList        = []
bidPriceList        = []
debug               = 1
obsSpace            = int(config['ENV']['ObservationSpace'])
initial_flag        = True 
old_data            = np.empty((0,2), float)
max_inventory       = float(config['ENV']['MaxInventory'])


print (ask_price_columns)

class securities_trading_env(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df):

        global bid_price_columns, ask_price_columns, starting_money,obsSpace

        super(securities_trading_env, self).__init__()

        self.df                 = df
        # self.reward_range       = (0, MAX_REWARD)
        self.MAX_STEPS          = len(df["L1bid_price"])  
        self.CURRENT_REWARD     = 0
        self.current_held_sec   = 0
        
        self.returns            = 0

        self.df_bidPrice        = df[df.columns[bid_price_columns]].transpose().values
        self.bid_history        = self.df_bidPrice.copy()

        self.df_askPrice        = df[df.columns[ask_price_columns]].transpose().values
        self.ask_history        = self.df_askPrice.copy()

        self.w0                 = np.array([1.0] + [0.0] * len(self.df_bidPrice))
        self.p0                 = 1.0
        self.netWorth           = 1.0
        self.trading_cost       = 0.


        print('TOTAL LIST---->'+str(len(bidPriceList)))
        print()

        self.action_space = spaces.Box(low=0., high=1., shape=(len(self.df_bidPrice) + 1,), dtype=np.float64)
        
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=((len(self.df_bidPrice) + 1) * obsSpace ,), dtype=np.float64)


    def step(self, action):

        global starting_money,obsSpace, Strategy
        # print(Strategy)

        # self._take_action(action)

        self.current_step += 1


        weights = np.clip(action, self.action_space.low, self.action_space.high)
        weights /= (np.sum(np.abs(weights)) + 1e-9)
        weights[0] += np.clip(1 - np.sum(np.abs(weights)), 0, 1)

        observation = self.df_bidPrice[:, self.current_step:self.current_step + obsSpace].copy()
        ask_observation = self.df_askPrice[:, self.current_step:self.current_step + obsSpace].copy()

        bias_observation = np.ones((1, obsSpace))


        '''
        print(np.shape(observation))
        print(np.shape(bias_observation))
        '''

        observation_with_bias = np.concatenate((bias_observation, observation), axis=0)

        ask_observation_with_bias = np.concatenate((bias_observation, ask_observation), axis=0)


        window = 10 if self.current_step > 10 else self.current_step - 1
        avg_now = self.df_bidPrice[:,:self.current_step].mean(axis=1,dtype = np.float64)
        avg_window = self.df_bidPrice[:,window:self.current_step].mean(axis=1,dtype = np.float64)
        

        mean_value = (avg_now[:]/avg_window[:])

        alpha = np.ones(observation_with_bias[:,-1].shape)

        if(Strategy == 'Momentum'):
            alpha = np.concatenate((np.ones(1,), mean_value), axis=0) 
        elif(Strategy == 'Bid-Ask'):
            alpha = ask_observation_with_bias[:,-1] / observation_with_bias[:,-1]
        elif(Strategy == 'Mean-Reversion'):
            mean_value = (avg_window[:]/avg_now[:])
            alpha = np.concatenate((np.ones(1,), mean_value), axis=0) 



        w1 = weights
        # print(w1.shape)
        assert w1.shape == alpha.shape, 'w1 and alpha must have the same shape'
        assert alpha[0] == 1.0, 'alpha[0] must be 1'

        w0 = self.w0
        p0 = self.p0

        dw1 = (alpha * w0) / (np.dot(alpha, w0) + 1e-9)  #  weights evolve into
        mu1 = self.trading_cost * (np.abs(dw1 - w1)).sum()  # cost to change portfolio
        
        assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = p0 * (1 - mu1) * np.dot(alpha, w1)  #  final portfolio value
        

        p1 = np.clip(p1, 0, np.inf)  # short not allowed

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + 1e-9) / (p0 + 1e-9))  # log rate of return
        reward = r1 *  (self.current_step/self.MAX_STEPS) # normalized logarithmic accumulated return
        
        # remember for next step
        self.w0 = w1
        self.p0 = p1

        self.netWorth = p1
        self.returns = rho1

        done = self.current_step >= self.MAX_STEPS or (p1 == 0)

        return observation_with_bias.reshape(-1), reward, done, {'return':self.returns,'portfolio value':self.netWorth}



    def reset(self):
        global askPriceList, bidPriceList, current_money, bid_price_columns, ask_price_columns, initial_flag, old_data,obsSpace

        self.CURRENT_REWARD     = 0
        self.current_step       = 0
        current_money           = starting_money
        df                      = self.df
        self.current_held_sec   = 0
        askPriceList            = []
        bidPriceList            = []
        initial_flag            = True
        old_data                = np.empty((0,2), float)
        self.netWorth           = starting_money


        
        self.w0 = np.array([1.0] + [0.0] * len(self.df_bidPrice))
        self.p0 = 1.0

        observation = self.df_bidPrice[:, self.current_step:self.current_step + obsSpace].copy() 
        bias_observation = np.ones((1, obsSpace))
        observation_with_bias = np.concatenate((bias_observation, observation), axis=0)
        
        return observation_with_bias.reshape(-1)


    def render(self, mode='human', close=False):
        global  current_money

        print(f'Step: {self.current_step}')
        # print(f'Price: {self.current_bidPrice}')
        print(f'Postfolio Value: {self.netWorth}')
        print(f'Returns: {self.returns}')
        # print(f'Current Reward: {self.CURRENT_REWARD}')
        # print(f'Net Worth: {self.netWorth}')