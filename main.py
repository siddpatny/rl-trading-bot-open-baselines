import gym
import json
import argparse
import datetime
import configparser 

import datetime as dt
import pandas as pd
import numpy as np

import logging
import random
import time
from collections import deque, namedtuple

import pika
import numpy as np

from trading_bots import test_pb2
from trading_bots import trading_account

from stable_baselines.common.policies 	import MlpPolicy
from stable_baselines.common.policies 	import MlpLstmPolicy
from stable_baselines.common.policies 	import MlpLnLstmPolicy
from stable_baselines.common.policies 	import CnnPolicy
from stable_baselines.common.policies 	import CnnLstmPolicy
from stable_baselines.common.policies 	import CnnLnLstmPolicy
from stable_baselines.common.vec_env 	import DummyVecEnv
from stable_baselines 					import PPO2
from stable_baselines import SAC
from stable_baselines.gail import generate_expert_traj

# from env.securities_trading_env 		import securities_trading_env
from env.sidd_trading_env 				import securities_trading_env

# Imports for DDPG
from stable_baselines 					import DDPG
from stable_baselines.ddpg.policies 	import MlpPolicy	as ddpgMlpPolicy
from stable_baselines.ddpg.noise 		import NormalActionNoise, OrnsteinUhlenbeckActionNoise,													   AdaptiveParamNoiseSpec
# Imports for TD3
from stable_baselines 					import TD3
from stable_baselines.td3.policies 		import MlpPolicy as td3MlpPolicy

# Imports for RLLib
# import ray
# from ray.rllib.agents.dqn import DQNTrainer
# from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
# from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
# from ray.rllib.agents.dqn.dqn import DQNTrainer
# # from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
# # from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
# from ray.tune.logger import pretty_print
# from ray.tune.registry import register_env


'''
Stable baselines framework
==========================

Policies:
---------
MlpPolicy		Policy object that implements actor critic, using a MLP (2 layers of 64)


Optimization algorithm:
------------------
PP02			Combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region 					to improve the actor).
TD3 			Is an off-policy algorithm. TD3 can only be used for environments with continuous 					action spaces.
DDPG 			Is an off-policy algorith that concurrently learns a Q-function and a policy. It is
				specifically adapted for continuous actions spaces.

'''


# episodes = 100
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

def train_gym(algo,args,env):
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

				for i in range(test_steps):
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
				    action, _states = model.predict(obs)
				    obs, rewards, done, info = env.step(action)
				    total_l += 1.
				    total_r += rewards
				    if done:
				    	break
				    env.render()

			# elif (algo == "GAIL"):
			# 	generate_expert_traj(model, env, n_timesteps=train_steps)
			# 	obs = env.reset()
			# 	for i in range(test_steps):
			# 	  action, _states = model.predict(obs)
			# 	  obs, rewards, dones, info = env.step(action)
			# 	  env.render()
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
	model_save = "save/"+algo+"-"+datenow+".h5"
	print("Model saved as: ",model_save)
	model.save(model_save)
	return model_save




class HitCounter(object):
	def __init__(self, seconds):
	    self.seconds = seconds
	    self.hits = [0] * seconds
	    self.times = [0] * seconds

	def hit(self, ts, count=1):
	    idx = ts % (self.seconds) if self.seconds != 0 else 0
	    if self.times[idx] == ts:
	        self.hits[idx] += count
	    else:
	        self.times[idx] = ts
	        self.hits[idx] = count
	    # print("HIT->"+str(ts)+"::"+str(self.seconds))

	def get(self, ts):
	    pairs = zip(self.times, self.hits)
	    return sum(h for t, h in pairs if ts - t < self.seconds)

TradeSide = namedtuple("TradeSide", ["buy", "sell"])

class call_back_Client:
	def __init__(self, num, window_size, agent, account, strategy, securities, host, limit):
		self.limit = limit
		self.account = account
		self.num = num  # number of securities that the bot listens to
		self.window_size = window_size  # number of orders for each security that the bot should receive to make an action
		self.agent = agent
		self.env = env  # the Agent object instance
		self.strategy = strategy  # strategy name, e.g. double_q_learning
		self.securities = securities  # list of security names
		self.sec = securities[0]  # current security
		self.sec2idx = dict()  # security to index mapping
		self.orders = set()  # list of unique external order no.
		self.prices = [0.0] * num  # list of market prices for each symb
		self.response = ""
		for i, sec in enumerate(securities):
		    self.sec2idx[sec] = i
		self.levels = [deque() for _ in range(self.num)]  # placeholder for input data
		seconds = window_size // 100
		self.trades = [
		    TradeSide(buy=HitCounter(seconds), sell=HitCounter(seconds))
		    for _ in securities
		]

		self.internalID = 0
		self.credentials = pika.PlainCredentials('test2', 'test2')
		self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, credentials=self.credentials))
		# self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
		self.channel = self.connection.channel()

		self.L1_queue = self.sec + "_L1_pb"
		self.L2_queue = self.sec + "_L2_pb"
		self.trade_queue = "trade_queue"
		self.ack_queue = "ack_queue"

		self.channel.exchange_declare(exchange="trade_data", exchange_type="direct")
		self.channel.queue_declare(queue=self.trade_queue)  # Keep queue name unique
		self.channel.queue_bind(
		    exchange="trade_data", queue=self.trade_queue, routing_key="trade_pb"
		)

		self.channel.exchange_declare(exchange="market_data_pb", exchange_type="direct")
		self.channel.queue_declare(queue=self.L2_queue)  # Keep queue name unique
		self.channel.queue_bind(
		    exchange="market_data_pb", queue=self.L2_queue, routing_key="l2_pb"
		)

		self.channel.exchange_declare(exchange="ACK", exchange_type="direct")
		self.channel.queue_declare(queue=self.ack_queue)  # Keep queue name unique
		self.channel.queue_bind(
		    exchange="ACK", queue=self.ack_queue, routing_key="ack_pb"
		)

		self.channel.basic_consume(
		    queue=self.L2_queue, on_message_callback=self.on_response_LL, auto_ack=True
		)

		# self.channel.basic_consume(
		#     queue=self.L1_queue, on_message_callback=self.on_response_L, auto_ack=True
		# )

		self.channel.basic_consume(
		    queue=self.trade_queue,
		    on_message_callback=self.on_response_trades,
		    auto_ack=True,
		)

		self.channel.basic_consume(
		    queue=self.ack_queue,
		    on_message_callback=self.on_response_ack,
		    auto_ack=True,
		)

	def on_response_LL(self, ch, method, properties, body):
		logging.debug("\n [x] Received Level 2")
		l2Data = test_pb2.L2Data()
		l2Data.ParseFromString(body)
		tob = {}
		tob["symb"] = str(l2Data.symb)
		tob["L1AskPrice"] = l2Data.L1AskPrice
		tob["L1BidPrice"] = l2Data.L1BidPrice
		tob["L2AskPrice"] = l2Data.L2AskPrice
		tob["L2BidPrice"] = l2Data.L2BidPrice
		tob["L3AskPrice"] = l2Data.L3AskPrice
		tob["L3BidPrice"] = l2Data.L3BidPrice
		tob["L4AskPrice"] = l2Data.L4AskPrice
		tob["L4BidPrice"] = l2Data.L4BidPrice
		tob["L5AskPrice"] = l2Data.L5AskPrice
		tob["L5BidPrice"] = l2Data.L5BidPrice
		tob["L1AskSize"] = l2Data.L1AskSize
		tob["L1BidSize"] = l2Data.L1BidSize
		tob["L2AskSize"] = l2Data.L2AskSize
		tob["L2BidSize"] = l2Data.L2BidSize
		tob["L3AskSize"] = l2Data.L3AskSize
		tob["L3BidSize"] = l2Data.L3BidSize
		tob["L4AskSize"] = l2Data.L4AskSize
		tob["L4BidSize"] = l2Data.L4BidSize
		tob["L5AskSize"] = l2Data.L5AskSize
		tob["L5BidSize"] = l2Data.L5BidSize
		# tob = (json.loads(body))  # todo replace json with protobuf
		logging.debug(tob)
		logging.debug(method.delivery_tag)
		if tob["symb"] in self.securities:
		    self.sec = tob["symb"]
		    logging.debug(
		        " [x] Received order with listening security: %s", tob["symb"]
		    )
		    # idx = self.securities.index(self.sec)
		    self.on_receive_required_symb(tob)
		else:
		    logging.debug(" [x] Ignored order for security: %s", tob["symb"])

	def on_response_L(self, ch, method, properties, body):  # consume callback
		logging.debug("\n [x] Received Level 1")
		# tob = (json.loads(body))  # todo replace json with protobuf
		tob = test_pb2.TOB()
		logging.debug(tob.ParseFromString(body))
		logging.debug("L1AskPrice: " + str(tob.L1AskPrice))
		logging.debug("L1BidPrice: " + str(tob.L1BidPrice))

	def on_response_trades(self, ch, method, properties, body):
		logging.debug("\n [x] Received trade")
		tradeobj = test_pb2.TradeOrder()
		tradeobj.ParseFromString(body)
		logging.debug("Response Trade:")
		logging.debug(
		    "buyOrderNo:"
		    + str(tradeobj.buyOrderNo)
		    + ",sellOrderNo:"
		    + str(tradeobj.sellOrderNo)
		    + ",tradeNo:"
		    + str(tradeobj.tradeNo)
		    + ",tradePrice:"
		    + str(tradeobj.tradePrice)
		    + ",tradeSize:"
		    + str(tradeobj.tradeSize)
		    + ",symbol:"
		    + str(tradeobj.symbol)
		)

		self.response = (
		    str(tradeobj.buyOrderNo)
		    + ","
		    + str(tradeobj.sellOrderNo)
		    + ","
		    + str(tradeobj.tradeNo)
		    + ","
		    + str(tradeobj.tradePrice)
		    + ","
		    + str(tradeobj.tradeSize)
		    + ","
		    + str(tradeobj.symbol)
		)

		if tradeobj.symbol not in self.securities:
		    return logging.debug(" [x] Ignored trade.")

		idx = self.sec2idx[tradeobj.symbol]
		if tradeobj.buyOrderNo:
		    counter = self.trades[idx].buy
		elif tradeobj.sellOrderNo:
		    counter = self.trades[idx].sell
		counter.hit(int(time.time()), tradeobj.tradeSize)

		# Update inventory and compute PnL
		if tradeobj.buyOrderNo in self.orders:
		    self.account.append(
		        tradeobj.tradeSize, tradeobj.tradePrice, tradeobj.symbol
		    )
		    logging.debug("Receive a buy trade")
		    self.orders.remove(tradeobj.buyOrderNo)
		if tradeobj.sellOrderNo in self.orders:
		    self.account.append(
		        -tradeobj.tradeSize, tradeobj.tradePrice, tradeobj.symbol
		    )
		    logging.debug("Receive a sell trade")
		    self.orders.remove(tradeobj.sellOrderNo)

		logging.debug("Current inventory is %s", self.account.get_inventory())
		logging.debug("Current cash balance is %s", self.account.get_balance())
		logging.debug(
		    "Current market value is %s", self.account.mark_to_market(self.prices)[0]
		)
		logging.debug(
		    "Current portfolio value (PnL) is %s",
		    self.account.mark_to_market(self.prices)[1],
		)

	def on_response_ack(self, ch, method, properties, body):  # record order no.
		logging.debug("\n [x] Received acknowledge")
		aMobj = test_pb2.aM()
		aMobj.ParseFromString(body)
		logging.debug("Response Ack:")
		logging.debug(
		    "strategy:"
		    + aMobj.strategy
		    + ",internalOrderNo:"
		    + str(aMobj.internalOrderNo)
		    + ",symb:"
		    + aMobj.symb
		    + ",orderNo:"
		    + str(aMobj.orderNo)
		    + ",action:"
		    + aMobj.action
		)
		if aMobj.action == "A":
		    self.orders.add(aMobj.orderNo)
		else:
		    self.orders.discard(aMobj.orderNo)

	def send_order(self, Order):
		# order = json.dumps(Order)
		order = test_pb2.OrderBody()
		order.symb = Order["symb"]
		order.price = Order["price"]
		order.origQty = Order["origQty"]
		order.orderNo = Order["orderNo"]
		order.status = Order["status"]
		order.remainingQty = Order["remainingQty"]
		order.action = Order["action"]
		order.side = Order["side"]
		order.FOK = Order["FOK"]
		order.AON = Order["AON"]
		order.strategy = Order["strategy"]

		self.channel.basic_publish(
		    exchange="",
		    routing_key=(self.sec + "_orders_pb"),
		    body=order.SerializeToString(),
		)
		logging.info(" [x] Sent order %s", Order["orderNo"])
		logging.info(order)

	def mystart(self):
		logging.info("Start consuming...")
		self.channel.start_consuming()

	def on_receive_required_symb(self, tob):
		idx = self.sec2idx[self.sec]
		queue = self.levels[idx]
		self.prices[idx] = 0.5 * (tob["L1BidPrice"] + tob["L1AskPrice"])
		logging.debug(" queue size %s", len(queue))
		if len(queue) >= self.window_size:
		    queue.popleft()  # maintain a queue of window_size to store latest orders
		queue.append(
		    [
		        tob["L1AskPrice"],
		        tob["L1AskSize"],
		        tob["L1BidPrice"],
		        tob["L1BidSize"],
		        tob["L2AskPrice"],
		        tob["L2AskSize"],
		        tob["L2BidPrice"],
		        tob["L2BidSize"],
		        tob["L3AskPrice"],
		        tob["L3AskSize"],
		        tob["L3BidPrice"],
		        tob["L3BidSize"],
		        tob["L4AskPrice"],
		        tob["L4AskSize"],
		        tob["L4BidPrice"],
		        tob["L4BidSize"],
		        tob["L5AskPrice"],
		        tob["L5AskSize"],
		        tob["L5BidPrice"],
		        tob["L5BidSize"],
		    ]
		)
		if all(
		    [len(q) == self.window_size for q in self.levels]
		):  # received enough data
		    logging.debug(" [x] Received enough data, start computing action...")
		    logging.debug(" [x] Received enough data, start computing action...")
		    orders = self.get_orders()
		    for order in orders:
		        self.send_order(order)
		        logging.debug(" [x] Sent a new order for action %s", order["action"])
		        # print(" [x] Sent a new order for action %s", order["action"])
		    else:
		    	# print(" [x] Performed no action for this state")
		        logging.debug(" [x] Performed no action for this state")

	def get_orders(self):
		now = int(time.time())
		levels = np.array([l[-1] for l in self.levels])
		# print("IN get orders")
		# print(levels)
		# print(levels.shape)
		logging.info("Levels: %s", levels.T[:,0])
		trades = np.array([[t.buy.get(now), t.sell.get(now)] for t in self.trades])
		inventory = self.account.inventory

		# print("old inventory")
		# print(inventory)
		# print(levels[:,0])
		logging.info("Old Inventory: %s", self.account.inventory)
		actions, _states = self.agent.predict(levels[:,0])
		
		# obs, rewards, done, info = env.step(new_inventory)
		print("predicted actions")
		print(np.sum(actions))
		new_inventory =  actions
		# new_inventory = np.random.uniform(low=-10, high=10, size=self.account.inventory.shape)
		new_inventory *= 10000
		new_inventory = np.round(new_inventory,0)
		# new_inventory -= 0.5
		# print("new inventory")
		
		# print(new_inventory)

		# if True: # trades.sum():
		# 	logging.info("Trades: %s", trades.T)
		# if True: # not np.array_equal(inventory, new_inventory):
		logging.info("New Inventory: %s", new_inventory)
		# if True:  # not np.array_equal(inventory, new_inventory):
		#     logging.info("New Inventory: %s", new_inventory)
		orders = []
		for idx, qty in enumerate(new_inventory):
			quantity = abs(new_inventory[idx] - inventory[idx])
			action = "A"
			if new_inventory[idx] > inventory[idx]:
			    side = "B"
			elif new_inventory[idx] < inventory[idx]:
			    side = "S"
			else:
			    continue

			orders.append(
			    {
			        "symb": self.securities[idx],
			        "price": self.prices[idx],
			        "origQty": quantity,
			        "orderNo": self.internalID,
			        "status": "A",
			        "remainingQty": quantity,
			        "action": action,
			        "side": side,
			        "FOK": 0,
			        "AON": 0,
			        "strategy": self.strategy,
			    }
			)
			self.internalID += 1
			if self.internalID > self.limit:
			    # self.connection.close()
			    print('Trading ended:')
			    print('Cash Balance: ', self.account.cash_balance)
			    print('Market Value: ', self.account.market_value)
			    print('Portfolio Value', self.account.portfolio_value)

			    self.channel.stop_consuming()
			    return []
		# print("orders:")
		# print(len(orders))
		return orders


if __name__ == "__main__":

	config = configparser.ConfigParser()
	config.read('config.ini')

	data_file 		= config['MAIN']['Data']
	test_steps 		= int(config['MAIN']['TestSize'])
	train_steps 	= int(config['MAIN']['TrainSize'])
	no_of_agents 	= int(config['MAIN']['BotNumber'])
	algo 			= str(config['MAIN']['Policy'])
	episodes = int(config['MAIN']['Episodes'])

	parser = argparse.ArgumentParser()


	#-p MlpLstmPolicy -a PP02 
	parser.add_argument("-l", "--load",  	dest = "loadFlag", 		default = "no_path", 	help="Only load the model")
	parser.add_argument("-v", "--verbose",  dest = "verboseFlag", 	default = 1, 			help="Flag for verbose either 1 or 0")
	parser.add_argument("--stop-reward", type=float, default=50)
	parser.add_argument("--limit", type=int, default=100)
	parser.add_argument("--torch", action="store_true")


	args = parser.parse_args()

	datenow = datetime.datetime.now().strftime("%I%M%p-%d%B%Y")

	print(args.loadFlag)

	# args = parser.parse_args()

	strategy = algo

	securities = ['ZFH0:MBO', 'ZTH0:MBO', 'UBH0:MBO', 'ZNH0:MBO', 'ZBH0:MBO',
	              'TNH0:MBO', 'GCG0:MBO', 'GEU2:MBO', 'GEM4:MBO', 'GEM2:MBO',
	              'GEM0:MBO', 'GEH0:MBO', 'GEZ0:MBO', 'GEZ1:MBO', 'GEM1:MBO',
	              'GEH4:MBO', 'GEU1:MBO', 'GEZ2:MBO', 'GEM3:MBO', 'GEH2:MBO',
	              'GEH1:MBO', 'GEH3:MBO', 'GEU0:MBO', 'GEZ3:MBO', 'GEU3:MBO']

	symb = [
	    "GEH0:MBO",
	    "GEH1:MBO",
	    "GEH2:MBO",
	    "GEM0:MBO",
	    "GEM1:MBO",
	    "GEM2:MBO",
	    "GEU0:MBO",
	    "GEU1:MBO",
	    "GEZ0:MBO",
	    "GEZ1:MBO",
	    "GEZ2:MBO",
	]
	num = len(symb)  # number of securities that the bot listens to
	window_size = 1000  # number of orders for each security that the bot should receive to make an action

	df = pd.read_csv(data_file)
	# env = DummyVecEnv([lambda: securities_trading_env(df)])
	env = DummyVecEnv([lambda: securities_trading_env(df)])

	saved_model = train_gym(algo,args, env)

	n_actions = env.action_space.shape[-1]
	action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
	model = DDPG(ddpgMlpPolicy, env, verbose=int(args.verboseFlag), param_noise=None, action_noise= action_noise)

	model.load(saved_model)
	# env.reset()

	# account = trading_account.Account(num, strategy, symb)
	# account.inventory = np.zeros(shape=(num))
	# # account.inventory = np.minimum(1000, account.inventory)
	# account.inventory = account.inventory.astype("float64")
	account = trading_account.Account(num, strategy, symb)
	account.inventory = np.random.normal(size=(num)) * 500
	account.inventory = np.minimum(1000, account.inventory)
	account.inventory = account.inventory.astype("int").astype("float64")

	my_call_back = call_back_Client(
		num, window_size, model, account, strategy, symb, "172.29.208.37", args.limit
	)
	my_call_back.mystart()








# def use_rllib(algo,args):
# 		ray.init()

# 		register_env("securities_trading_env",lambda _: securities_trading_env(df))
# 		env = SubprocVecEnv([make_env(env_id, log_dir, i+worker_id) for i in range(num_env)])
# 		single_env = DummyVecEnv([lambda: securities_trading_env(df)])
# 		obs_space = single_env.observation_space
# 		act_space = single_env.action_space

# 		# You can also have multiple policies per trainer, but here we just
# 		# show one each for PPO and DQN.
# 		policies = {
# 		"ppo_policy": (PPOTFPolicy, obs_space, act_space, {}),
# 		"dqn_policy": (DQNTFPolicy , obs_space, act_space, {}),
# 		}

# 		def policy_mapping_fn(agent_id):
# 			if agent_id % 2 == 0:
# 			    return "ppo_policy"
# 			else:
# 			    return "dqn_policy"

# 		ppo_trainer = PPOTrainer(
# 		env="securities_trading_env",
# 		config={
# 		    "multiagent": {
# 		        "policies": policies,
# 		        "policy_mapping_fn": policy_mapping_fn,
# 		        "policies_to_train": ["ppo_policy"],
# 		    },
# 		    "explore": False,
# 		    # disable filters, otherwise we would need to synchronize those
# 		    # as well to the DQN agent
# 		    "observation_filter": "NoFilter",
# 		    "use_pytorch": args.torch,
# 		})

# 		dqn_trainer = DQNTrainer(
# 		env="multi_agent_cartpole",
# 		config={
# 		    "multiagent": {
# 		        "policies": policies,
# 		        "policy_mapping_fn": policy_mapping_fn,
# 		        "policies_to_train": ["dqn_policy"],
# 		    },
# 		    "gamma": 0.95,
# 		    "n_step": train_steps,
# 		    "use_pytorch": args.torch,
# 		})

# 		# You should see both the printed X and Y approach 200 as this trains:
# 		# info:
# 		#   policy_reward_mean:
# 		#     dqn_policy: X
# 		#     ppo_policy: Y
# 		for i in range(episodes):
# 			print("== Iteration", i, "==")

# 			# improve the DQN policy
# 			print("-- DQN --")
# 			result_dqn = dqn_trainer.train()
# 			print(pretty_print(result_dqn))

# 			# improve the PPO policy
# 			print("-- PPO --")
# 			result_ppo = ppo_trainer.train()
# 			print(pretty_print(result_ppo))

# 			# Test passed gracefully.
# 			# if args.as_test and \
# 			#         result_dqn["episode_reward_mean"] > args.stop_reward and \
# 			#         result_ppo["episode_reward_mean"] > args.stop_reward:
# 			#     print("test passed (both agents above requested reward)")
# 			#     quit(0)

# 			# swap weights to synchronize
# 			dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
# 			ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))

    # Desired reward not reached.
    # if args.as_test:
    #     raise ValueError("Desired reward ({}) not reached!".format(
    #         args.stop_reward))

# use_rllib(algo, args)
			
