import random
import gym
import math
import configparser 
import pandas   as pd
import numpy    as np
from   gym      import spaces






# print (ask_price_columns)

class securities_trading_env(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, priceList):

        super(securities_trading_env, self).__init__()

        config = configparser.ConfigParser()
        config.read('config.ini')


        self.strategy            = str(config['ENV']['Strategy'])
        self.starting_money      = float(config['ENV']['StartingMoney'])

        self.current_money       = self.starting_money

        

        self.priceList       = priceList.T
        # self.reward_range       = (0, MAX_REWARD)
        self.max_steps          = len(self.priceList[0])
        self.current_step       = 0
        self.returns            = 0

        self.w0                 = np.array([0.0] * len(self.priceList))
        self.p0                 = 1.0
        self.netWorth           = self.starting_money
        # self.out_of_money       = False
        self.trading_cost       = 0.


        print('TOTAL LIST---->'+str(self.priceList.shape))
        # print(self.df_bidPrice[0])

        self.action_space = spaces.Box(low=0., high=1., shape=(len(self.priceList),), dtype=np.float64)
        
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=((len(self.priceList)),), dtype=np.float64)


    def step(self, action):

        # print(Strategy)

        # self._take_action(action)

        self.current_step += 1

        # print("ACTION:")
        # print(action.shape)
        # print(action)
        weights = np.clip(action, self.action_space.low, self.action_space.high)
        # weights = action
        weights /= (np.sum(np.abs(weights)) + 1e-9)
        # weights[0] += np.clip(1 - np.sum(np.abs(weights)), 0, 1)
        # print(weights)
        # print(np.sum(weights))
        # weights = action

        curr_observation = self.priceList[:, self.current_step:self.current_step + 1].copy()
        # ask_observation = self.df_askPrice[:, self.current_step:self.current_step + 1].copy()
        # print("observation")
        # print(observation.shape)

        # bias_observation = np.ones((1, obsSpace))


        '''
        print(np.shape(observation))
        print(np.shape(bias_observation))
        '''

        # observation_with_bias = np.concatenate((bias_observation, observation), axis=0)

        # ask_observation_with_bias = np.concatenate((bias_observation, ask_observation), axis=0)


        window = 50 if self.current_step > 50 else 0
        # print(self.priceList[:,window:self.current_step])
        avg_now = self.priceList[:,:self.current_step].mean(axis=1,dtype = np.float64)
        avg_window = self.priceList[:,window:self.current_step].mean(axis=1,dtype = np.float64)
        

        mean_value = (avg_now[:]/avg_window[:])

        alpha = np.ones(curr_observation[:,-1].shape)

        if(self.strategy == 'Momentum'):
            alpha = mean_value 
        # elif(Strategy == 'Bid-Ask'):
        #     alpha = ask_observation[:,-1] / observation[:,-1]
        elif(self.strategy == 'Mean-Reversion'):
            mean_value = (avg_window[:]/avg_now[:])
            alpha = mean_value 



        w1 = weights
        # print("Weights")
        # print(w1.shape)
        # print(w1)

        # print("Alpha")
        # print(alpha.shape)
        # print(alpha)
        assert w1.shape == alpha.shape, 'w1 and alpha must have the same shape'
        # assert alpha[0] == 1.0, 'alpha[0] must be 1'

        w0 = self.w0
        p0 = self.p0

        # dw1 = (alpha * w0) / (np.dot(alpha, w0) + 1e-9)  #  weights evolve into
        # mu1 = self.trading_cost * (np.abs(dw1 - w1)).sum()  # cost to change portfolio
        # assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = p0 * np.dot(alpha, np.abs(w1))  #  final portfolio value
        # p1 /= (np.sum(np.abs(p1)) + 1e-9)

        # print("Portfolio")
        # print(p1.shape)
        # print(p1)

        # p1 = np.clip(p1, 0, np.inf)  # short not allowed

        rho1 = (p1 / (p0+1e-9)) - 1  # rate of returns
        r1 = (p1 + 1e-9) / (p0 + 1e-9) # rate of return
        reward = p1*self.starting_money *  (self.current_step/self.max_steps) # normalized logarithmic accumulated return
        
        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        # self.out_of_money = (p1 == 0)

        self.current_money = p1*self.starting_money
        self.returns = rho1

        done = self.current_step >= self.max_steps

        return curr_observation.reshape(-1), reward, done, {'return':self.returns,'portfolio value':self.netWorth}



    def reset(self):

        self.current_step       = 0
        self.netWorth           = self.starting_money
        self.returns            = 0

        self.w0 = np.array([0.0] * len(self.priceList))
        self.p0 = 1.0

        curr_observation = self.priceList[:, 0].copy() 
        
        return curr_observation.reshape(-1)


    def render(self, mode='human', close=False):

        print(f'Step: {self.current_step}')
        # print(f'Price: {self.current_bidPrice}')
        print(f'Postfolio Value: {self.netWorth}')
        print(f'Returns: {self.returns}')
        # print(f'Current Reward: {self.CURRENT_REWARD}')
        # print(f'Net Worth: {self.netWorth}')