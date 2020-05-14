# rl-trading-bot-open-baselines
1) Create a Portfolio of Stocks using a reinforcement agent from Open AI gym's stable baselines.
2) Experiment with different trading strategies.
3) Deploy multiple agents with RLLib.

## Dataset Description
Folder: /data/concat.csv </br>
Its a static dataset consisting of the bid_price,ask_price,bid_size,ask_size for 25 securities for 1000 timesteps

# Requirements
tensorflow 1.x, PyTorch 1.5, Mpi4py, cuda

## Configuration
File: config.ini
Look out for:
1) Baseline Algorithm: (DDPG/TD3/PPO2)
2) Episodes: No of epochs to train
3) Strategy: Trading Strategy to implement
4) Train,Test Size


## Algorithms
RL Algorithms and Policy:
1) DDPG - DDPGMLP 
2) TD3 - TD3MLP
3) PPO2 - MLPLSTM </br>
These alogrithms are imported from stable baselines [[1]](#1) and trained in a custom Open AI Gym env [[2]](#2).

## Custom Env
Folder: /env
1) (Default) sidd_trading_env: My custom gym env for custom trading strategy for 25 securities.
2) securities_trading_env: Single security trading algorithm with discrete actions - buy, sell, hold.

Observation Space: Box - current bid_price of 25 securities + bias (26,) </br>
Action Space: Box - 26 row vector containing weights of for the 25 securities + bias term in the range [0,1] (Only longs or reallocation, no shorts (negative weights))

## Trading Strategies
1) Momentum: Ratio of the average bid price in the window with the average price upto current step.
2) Ask/Bid Ratio: Takes a long position weighted by the bid ask spread.
3) Mean Reversion: Inverse of momentum. Assumes the security is mean reverting.

The current action defines the weights of the portfolio. The sample from the action is clipped between 0,1 and normalized such that the sum of all the weights = 1. This ensures that the portfolio is completely utilized with a distribution of securities. (Only longs) </br>
The reward for the action is log rate of return with the new weights of the portfolio normalized by the progress for a delayed reward.

## To Fix RLLib
RLLib [[7]](#7) is another scalable reinforcement library that I want to use for 2 main purposes:
1) Hyper-parameter tuning [[3]](#3) for the model. An example can be seen in [[4]](#4)
2) Multiple Agents: This can be done in two ways:
  a) Train two different models and policies and sync them peiodically. [[5]](#5)
  b) Train different strategies in an adverserial manner. [[6]](#6)

Currently getting issues during configuration of multiple agents running on the same open ai custom env on different nodes.


## Future Work
1) Convert action into discrete buy,sell,hold based on portfolio weights and orders with bid,ask sizes
2) Connect to RabbitMQ for stream of orders


## References
<a id="1">[1]</a> 
https://stable-baselines.readthedocs.io/en/master/

<a id="2">[2]</a>
https://gym.openai.com/

<a id="3">[3]</a>
https://docs.ray.io/en/master/tune.html

<a id="4">[4]</a>
https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_ppo_example.py

<a id="5">[5]</a>
https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_two_trainers.py

<a id="6">[6]</a>
https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py

<a id="7">[7]</a>
https://github.com/ray-project/ray
