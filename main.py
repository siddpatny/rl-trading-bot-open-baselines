import gym
import json
import argparse
import datetime
import configparser 

import datetime as dt
import pandas as pd
import numpy as np

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
import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.dqn.dqn import DQNTrainer
# from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
# from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env


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

config = configparser.ConfigParser()
config.read('config.ini')

data_file 		= config['MAIN']['Data']
test_steps 		= int(config['MAIN']['TestSize'])
train_steps 	= int(config['MAIN']['TrainSize'])
no_of_agents 	= int(config['MAIN']['BotNumber'])
algo 			= str(config['MAIN']['Policy'])
episodes = int(config['MAIN']['Episodes'])

parser = argparse.ArgumentParser()
df = pd.read_csv(data_file)
# env = DummyVecEnv([lambda: securities_trading_env(df)])
env = DummyVecEnv([lambda: securities_trading_env(df)])

#-p MlpLstmPolicy -a PP02 
parser.add_argument("-l", "--load",  	dest = "loadFlag", 		default = "no_path", 	help="Only load the model")
parser.add_argument("-v", "--verbose",  dest = "verboseFlag", 	default = 1, 			help="Flag for verbose either 1 or 0")
parser.add_argument("--stop-reward", type=float, default=50)
parser.add_argument("--torch", action="store_true")


args = parser.parse_args()

datenow = datetime.datetime.now().strftime("%I%M%p-%d%B%Y")

print(args.loadFlag)
# episodes = 100


def use_gym(algo,args):
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
		model = PPO2(MlpPolicy, env, verbose=int(args.verboseFlag))


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

			elif (algo == "GAIL"):
				generate_expert_traj(model, env, n_timesteps=train_steps)
				obs = env.reset()
				for i in range(test_steps):
				  action, _states = model.predict(obs)
				  obs, rewards, dones, info = env.step(action)
				  env.render()
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
				# model = DDPG(ddpgMlpPolicy, env, verbose=int(args.verboseFlag), param_noise=None, action_noise= NormalActionNoise)
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

use_gym(algo,args)


def use_rllib(algo,args):
		ray.init()

		register_env("securities_trading_env",lambda _: securities_trading_env(df))
		env = SubprocVecEnv([make_env(env_id, log_dir, i+worker_id) for i in range(num_env)])
		single_env = DummyVecEnv([lambda: securities_trading_env(df)])
		obs_space = single_env.observation_space
		act_space = single_env.action_space

		# You can also have multiple policies per trainer, but here we just
		# show one each for PPO and DQN.
		policies = {
		"ppo_policy": (PPOTFPolicy, obs_space, act_space, {}),
		"dqn_policy": (DQNTFPolicy , obs_space, act_space, {}),
		}

		def policy_mapping_fn(agent_id):
			if agent_id % 2 == 0:
			    return "ppo_policy"
			else:
			    return "dqn_policy"

		ppo_trainer = PPOTrainer(
		env="securities_trading_env",
		config={
		    "multiagent": {
		        "policies": policies,
		        "policy_mapping_fn": policy_mapping_fn,
		        "policies_to_train": ["ppo_policy"],
		    },
		    "explore": False,
		    # disable filters, otherwise we would need to synchronize those
		    # as well to the DQN agent
		    "observation_filter": "NoFilter",
		    "use_pytorch": args.torch,
		})

		dqn_trainer = DQNTrainer(
		env="multi_agent_cartpole",
		config={
		    "multiagent": {
		        "policies": policies,
		        "policy_mapping_fn": policy_mapping_fn,
		        "policies_to_train": ["dqn_policy"],
		    },
		    "gamma": 0.95,
		    "n_step": train_steps,
		    "use_pytorch": args.torch,
		})

		# You should see both the printed X and Y approach 200 as this trains:
		# info:
		#   policy_reward_mean:
		#     dqn_policy: X
		#     ppo_policy: Y
		for i in range(episodes):
			print("== Iteration", i, "==")

			# improve the DQN policy
			print("-- DQN --")
			result_dqn = dqn_trainer.train()
			print(pretty_print(result_dqn))

			# improve the PPO policy
			print("-- PPO --")
			result_ppo = ppo_trainer.train()
			print(pretty_print(result_ppo))

			# Test passed gracefully.
			# if args.as_test and \
			#         result_dqn["episode_reward_mean"] > args.stop_reward and \
			#         result_ppo["episode_reward_mean"] > args.stop_reward:
			#     print("test passed (both agents above requested reward)")
			#     quit(0)

			# swap weights to synchronize
			dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
			ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))

    # Desired reward not reached.
    # if args.as_test:
    #     raise ValueError("Desired reward ({}) not reached!".format(
    #         args.stop_reward))

# use_rllib(algo, args)
			
