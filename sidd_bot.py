import random
import pika
import time
import pandas as pd
import numpy as np
import json
import test_pb2
import trading_account
# import double_q_learning_multiple
import collections
from collections import deque
import datetime
# import get_q_learning_agent
import sys

import datetime as dt
# import pandas as pd
import numpy as np

import logging
import random
import time
from collections import deque, namedtuple

import gym
import json
import argparse
import datetime
import configparser 


from stable_baselines.common.policies   import MlpPolicy
from stable_baselines.common.policies   import MlpLstmPolicy
from stable_baselines.common.policies   import MlpLnLstmPolicy
from stable_baselines.common.policies   import CnnPolicy
from stable_baselines.common.policies   import CnnLstmPolicy
from stable_baselines.common.policies   import CnnLnLstmPolicy
from stable_baselines.common.vec_env    import DummyVecEnv,VecCheckNan
from stable_baselines                   import PPO2
from stable_baselines import SAC
from stable_baselines.gail import generate_expert_traj

# from env.securities_trading_env       import securities_trading_env
from env.sidd_trading_env               import securities_trading_env

# Imports for DDPG
from stable_baselines                   import DDPG
from stable_baselines.ddpg.policies     import MlpPolicy    as ddpgMlpPolicy
from stable_baselines.ddpg.noise        import NormalActionNoise, OrnsteinUhlenbeckActionNoise,                                                    AdaptiveParamNoiseSpec
# Imports for TD3
from stable_baselines                   import TD3
from stable_baselines.td3.policies      import MlpPolicy as td3MlpPolicy

from mx_communication import Communication


# logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

def train_gym(algo,args,env,train_steps,test_steps):
    ep_r = []
    ep_l = []
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    print(n_actions)
    if(algo == "DDPG"):
        model = DDPG(ddpgMlpPolicy, env, verbose=int(args.verboseFlag), param_noise=param_noise, action_noise= action_noise)
    elif(algo=="TD3"):
        model = TD3(td3MlpPolicy, env, verbose=int(args.verboseFlag))
    elif(algo=="GAIL"):
        model = TD3(td3MlpPolicy, env, verbose=int(args.verboseFlag))
    else:
        model = PPO2(MlpLnLstmPolicy, env, verbose=int(args.verboseFlag))


    for e in range(episodes):
        print("EPISODE====>" + str(e))
        # obs = env.reset()
        # obs = env.reset()
        total_r = 0.
        total_l = 0.

        if (args.loadFlag == "no_path"):
            if (algo == "PPO2"):
                # model = PPO2(MlpPolicy, env, verbose=int(args.verboseFlag))
                model.learn(total_timesteps=train_steps)
                obs = env.reset()
                # print(obs)
                for i in range(test_steps):
                    print(obs)
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    total_l += 1.
                    total_r += rewards
                    if done:
                        break
                    env.render()


            elif (algo == "TD3"):
                # model = TD3(td3MlpPolicy, env, verbose=int(args.verboseFlag))
                model.learn(total_timesteps=train_steps, log_interval=10)
                obs = env.reset()

                for i in range(test_steps):
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    total_l += 1.
                    total_r += rewards
                    if done:
                        break
                    env.render()

                # model_save = "save/TD3"+"-"+datenow+".h5"
                # print("Model saved as: ",model_save)
                # model.save(model_save)
            
            elif (algo == "DDPG"):
                # model = DDPG(ddpgMlpPolicy, env, verbose=int(args.verboseFlag), param_noise=None, action_noise= None)
                model.learn(total_timesteps=train_steps)
                obs = env.reset()

                for i in range(test_steps):
                    # print(obs)
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    total_l += 1.
                    total_r += rewards
                    if done:
                        break
                    env.render()

            # elif (algo == "GAIL"):
            #   generate_expert_traj(model, env, n_timesteps=train_steps)
            #   obs = env.reset()
            #   for i in range(test_steps):
            #     action, _states = model.predict(obs)
            #     obs, rewards, dones, info = env.step(action)
            #     env.render()
                # model_save = "save/DDPG"+"-"+datenow+".h5"
                # print("Model saved as: ",model_save)
                # model.save(model_save)

        else:
            if (algo == "PPO2"):
                model = PPO2(MlpPolicy, env, verbose=int(args.verboseFlag))
                model.load(args.loadFlag)
                obs = env.reset()

                for i in range(test_steps):
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    env.render()

            elif (algo == "TD3"):
                model = TD3(td3MlpPolicy, env, verbose=int(args.verboseFlag))
                model.load(args.loadFlag)
                obs = env.reset()

                for i in range(test_steps):
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    env.render()
            
            elif (algo == "DDPG"):
                model = DDPG(ddpgMlpPolicy, env, verbose=int(args.verboseFlag), param_noise=None, action_noise= NormalActionNoise)
                model.load(args.loadFlag)
                obs = env.reset()

                for i in range(test_steps):
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    total_l += 1.
                    total_r += rewards
                    if done:
                        break
                    env.render()

        ep_r.append(total_r)
        ep_l.append(total_l)

    print("episode mean reward: {:0.3f} mean length: {:0.3f}".format(np.mean(ep_r), np.mean(ep_l)))
    # model_save = "save/PPO2"+"-"+datenow+".h5"
    model_save = "save/large1"+algo+"-"+datenow+".h5"
    print("Model saved as: ",model_save)
    model.save(model_save)
    return model_save

class management(Communication):
    
    def _init_sec_prices(self, securities):
        sec_state = dict()
        for sec in securities:
            sec_state.setdefault(sec, None)
        return sec_state

    def _init_market_dict(self, market_event_securities, market_event_queue):
        market_dict = dict()
        for sec in market_event_securities:
            sym_dict = dict()
            for e in market_event_queue:
                sym_dict[e] = None
            market_dict[sec] = sym_dict
        return market_dict
    
    # size of each security hold is set to be 0 initially
    def _init_inventory(self, securities):
        inventory = dict()
        for sec in securities:
            inventory[sec] = 0.0
        return inventory

    def __init__(self, market_event_securities , market_event_queue, securities, queue, host,
                    strategy,model,env,limit,window_size):

        # env = DummyVecEnv([lambda: securities_trading_env(df)])
        # model = TD3(td3MlpPolicy, env, verbose=int(args.verboseFlag))
        self.strategy = strategy # identifier for different clients 
        self.market_event_securities = market_event_securities # strings of securities, e.g. [ZFH0:MBO,ZTH0:MBO,UBH0:MBO,ZNH0:MBO,ZBH0:MBO]
        self.market_event_queue = market_event_queue # strings of names of prices in market_event_securities, e.g. [L1, L2, L3]
        self.securities = securities # strings of securities that can be traded in e.g [ZFH0:MBO,ZTH0:MBO]
        self.queue = queue # strings of names of prices in securities, e.g. [L1,L2]
        self.model = model
        self.limit = limit
        self.window_size = window_size
        self.num_of_securities = len(self.securities) # number of securities the bot will trade in
        self.internalID = 0 # internal id for every order the bot wants to send
        self.steps = 0 # number of trades the bot has made
        self.observations = []
        self.env = env
        self.cash_balance = 10000.0
        self.inventory = self._init_inventory(self.securities) # size of each security hold
        self.inventoryValue = 0.0
        self.PnL = self.cash_balance + self.inventoryValue
        
        
        self.market_dict = self._init_market_dict(self.market_event_securities,  self.market_event_queue) # L1-L5 levels data
        # self.market_dict["ZTH0:MBO"]["L1"] to read l1 data of ZTH0:MBO
        self.mid_market = self._init_sec_prices(securities) # half of the sum of current L1 ask price and L1 bid price
        self.exIds_to_inIds = dict() # when your order is acked, the bot will receive an external id for it. map exid to inid here.
        self.inIds_to_orders_sent = dict() # orders sent but not acked
        self.inIds_to_orders_confirmed = dict() # orders confirmed by matching agent


        self.talk = Communication(market_event_securities, market_event_queue, securities, queue, host,
                                  callback_for_levels = self.callback_for_levels,
                                  callback_for_acks = self.callback_for_acks,
                                  callback_for_trades = self.callback_for_trades,
                                  strategy = self.strategy)
        self.talk.kickoff()
    
    def _save_order_being_sent(self, order):
        self.inIds_to_orders_sent[order["orderNo"]] = order
   
    def cancel_order(self, order):
        self.talk._cancel_order(order)

    def send_order(self, order):
        self._save_order_being_sent(order)
        self.talk._send_order(order)

    def _update_with_trade(self, tradeobj, side, exId):
        # buy side = 1, sell side = -1
        self._update_inventory(tradeobj.symbol, tradeobj.tradeSize * side)
        self._update_cash(tradeobj.tradeSize, tradeobj.tradePrice * (-side))
        self._update_pnl()
        self._update_order_remain(exId, tradeobj.tradeSize)
        print('trade updated')
        sys.exit()
    
    def _update_inventory(self, symbol, size):
        self.inventory[symbol] += size
        inventoryValue = 0.0
        for sec in self.securities:
            inventoryValue += self.inventory[sec] * self.mid_market[sec]
        self.inventoryValue = inventoryValue

    def _update_cash(self, size, price):
        self.cash_balance += size * price
    
    def _update_pnl(self,):
        self.PnL = self.cash_balance + self.inventoryValue
    
    def _update_order_remain(self, exId, size):
        inId = self.exIds_to_inIds[exId]
        self.inIds_to_orders_confirmed[inId]["remainingQty"] -= size
        if self.inIds_to_orders_confirmed[inId]["remainingQty"] == 0:
            self.inIds_to_orders_confirmed.pop(inId)

    # only accept trade which belongs to this bot
    def _condition_to_accept_trade(self, tradeobj):
        exId = 0
        if tradeobj.buyOrderNo in list(self.exIds_to_inIds.keys()):
            return tradeobj.buyOrderNo, 1
        elif tradeobj.sellOrderNo in list(self.exIds_to_inIds.keys()):
            return tradeobj.sellOrderNo, -1
        else:
            return exId, 0

    def callback_for_trades(self, tradeobj):
        exId, side = self._condition_to_accept_trade(tradeobj)
        if side == -1 or side == 1:
            # uodate inventory, pnl, manage orders, decrease reamaining qty, if reamaining qty is 0, remove it from orders_confirmed
            self._update_with_trade(tradeobj, side, exId)
            self.steps = self.steps+1
            
            #self._model_reaction_to_trade(tradeobj)

    ### depends on different models
    def _model_reaction_to_trade(self,tradeobj):
        pass

    def _update_with_ack(self, inId, exId):
        self.exIds_to_inIds[exId] = inId
        print("ExId: %s -> InId: %s" % (exId, inId))
        if aMobj.action == "A":
            self.inIds_to_orders_confirmed[inId] = self.inIds_to_orders_sent.pop(inId)
        else:
            self.inIds_to_orders_sent[inId] = self.inIds_to_orders_confirmed.pop(inId)

    #record orders which are not successfully sent or canceled in case you want to send them again and map exid to inid
    def callback_for_acks(self, aMobj):
        if (aMobj.strategy  == self.strategy):
            self._update_with_ack(aMobj.internalOrderNo, aMobj.orderNo)

            self._model_reaction_to_ack(aMobj)

    ### depends on different models
    def _model_reaction_to_ack(self, aMobj):
        print("In ACK")
        print(aMobj)
        # pass

    def _update_market_dict(self, tob):
        for lv in self.talk.all_queue:
            self.market_dict[tob["symb"]][lv] = {lv+"AskPrice":tob[lv+"AskPrice"],
                                                 lv+"BidPrice":tob[lv+"BidPrice"],
                                                 lv+"AskSize":tob[lv+"AskSize"],
                                                 lv+"BidSize":tob[lv+"BidSize"]}
        self.mid_market[tob["symb"]] = 0.5 * (self.market_dict[tob["symb"]]["L1"]["L1AskPrice"] + self.market_dict[tob["symb"]]["L1"]["L1BidPrice"])

    # should be called when new level data arrives
    def callback_for_levels(self, tob):
        self._update_market_dict(tob)
        self._model_reaction_to_level(tob)

    def _model_reaction_to_level(self, tob):
        # if self.internalID > self.limit:
        #     return
        
        
        observation = np.array([v for v in self.mid_market.values()])
        if(self._condition_to_make_prediction() and observation.size == len(self.securities)):
            if(len(self.observations) == self.window_size+1):
                print("Model reaction")
                env = DummyVecEnv([lambda: securities_trading_env(np.array(self.observations).T,num)])
                self.model.learn(total_timesteps=self.window_size)
                self.env.reset()
                actions,_states = self.model.predict(observation)
                self._generate_orders(actions)
                self.observations = []
                env.render()
            else:
                self.observations.append(observation)

        # print("observation")
        # print(self.mid_market)
        # print(observation)
        # actions, _states = self.model.predict(observation)
    def _generate_orders(actions):
        print("action")
        print(actions)
        print("inventory")
        print(self.inventory)
        if self._condition_to_make_prediction():
            new_inventory =  actions
            # new_inventory = np.random.uniform(low=-10, high=10, size=self.account.inventory.shape)
            new_inventory *= 10000
            new_inventory = np.round(new_inventory,0)
            orders = []
            for idx, sec in enumerate(self.mid_market.keys()):
                quantity = abs(new_inventory[idx] - self.inventory[sec])
                action = "A"
                if new_inventory[idx] > self.inventory[sec]:
                    side = "B"
                elif new_inventory[idx] < self.inventory[sec]:
                    side = "S"
                else:
                    continue

                orders.append(
                    {
                        "symb": sec,
                        "price": self.mid_market[sec],
                        "origQty": 5,
                        "status": "A",
                        "remainingQty": 5,
                        "orderNo": self.internalID,
                        "action": action,
                        "side": side,
                        "FOK": 0,
                        "AON": 0,
                        "strategy": self.strategy,
                    }
                )

                self.internalID += 1
            # self.send_order(orders)
            if self.internalID <= self.limit:
                for order in orders:
                    self.send_order(order)
                # self.connection.close()

               



    
    # make adaption accordingly, e.g. double q only makes prediction when each symbols has a window of prices
    def _condition_to_make_prediction(self,):
        return not (None in self.mid_market.values())
    '''
    # transform prices into model input
    def _next_observation(self,):
        # for gym, maybe it can be called like return env._next_observation(self.mid_market)
        return self.mid_market
    '''



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    data_file       = config['MAIN']['Data']
    test_steps      = int(config['MAIN']['TestSize'])
    train_steps     = int(config['MAIN']['TrainSize'])
    no_of_agents    = int(config['MAIN']['BotNumber'])
    algo            = str(config['MAIN']['Policy'])
    episodes = int(config['MAIN']['Episodes'])

    parser = argparse.ArgumentParser()


    #-p MlpLstmPolicy -a PP02 
    parser.add_argument("--load",     dest = "loadFlag",      default = "no_path",    help="Only load the model")
    parser.add_argument("-v", "--verbose",  dest = "verboseFlag",   default = 1,            help="Flag for verbose either 1 or 0")
    parser.add_argument("--stop-reward", type=float, default=50)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--torch", action="store_true")


    args = parser.parse_args()

    datenow = datetime.datetime.now().strftime("%I%M%p-%d%B%Y")

    print(args.loadFlag)

    # args = parser.parse_args()

    strategy = algo

    market_event_securities = ["ZFH0:MBO","ZTH0:MBO","UBH0:MBO","ZNH0:MBO","ZBH0:MBO"]
    market_event_queue = ["L1","L2","L3"]
    securities = ["ZTH0:MBO", "UBH0:MBO"]
    # securities = market_event_securities
    num = len(securities)  # number of securities that the bot listens to
    window_size = 10  # number of orders for each security that the bot should receive to make an action

    df = pd.read_csv(data_file)
    bid_price_columns   = list(map(int,config['ENV']['ColumnsOfBidPrice'].split(',')))
    ask_price_columns   = list(map(int,config['ENV']['ColumnsOfAskPrice'].split(',')))

    bidPrices        = df[df.columns[bid_price_columns[-num:]]].transpose().values

    askPrices        = df[df.columns[ask_price_columns[-num:]]].transpose().values

    # mid_data = np.array([(x + y) / 2 for x, y in zip(askPrices, bidPrices)]).T
    # for i in range(bidPrices):
    # env = DummyVecEnv([lambda: securities_trading_env(df)])
    print('bid prices')
    print(bidPrices.shape)
    env = DummyVecEnv([lambda: securities_trading_env(np.array(bidPrices).T,num)])
    env = VecCheckNan(env, raise_exception=True)

    saved_model = train_gym(algo,args, env,train_steps,test_steps)

    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model = DDPG(ddpgMlpPolicy, env, verbose=int(args.verboseFlag), param_noise=None, action_noise= action_noise)

    model.load(saved_model)

    queue = ["L1","L2"]
    # host = "localhost"
    host = "172.29.208.37"
    # host = "172.29.212.100"
    my_interface = management(market_event_securities, market_event_queue, securities, queue, host, strategy,model,env,args.limit,window_size)
    #print('Interface Set Up!')



        



