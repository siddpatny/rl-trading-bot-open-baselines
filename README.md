# rl-trading-bot-open-baselines
Create a Portfolio of Stocks using Open AI gym's stable baselines

## Dataset Description
Folder: /data/concat.csv
Its a static dataset consisting of the bid_price,ask_price,bid_size,ask_size for 25 securities for 1000 timesteps

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
3) PPO2 - MLPLSTM
These alogrithms are imported from stable baselines [[1]](#1) and trained in a custom Open AI Gym env [[2]](#2).

## Custom Env
Folder: /env
1) (Default) sidd_trading_env: My custom gym env for custom trading strategy for 25 securities.
2) securities_trading_env: Single security trading algorithm with discrete actions - buy, sell, hold.

## Trading Strategies
1) Momentum: Ratio of the average bid price in the window with the average price upto current step.
2) Ask/Bid Ratio: Takes a long position weighted by the bid ask spread.
3) Mean Reversion: Inverse of momentum. Assumes the security is mean reverting.

## RLLib 


## Future Work
1) Convert action into discrete buy,sell,hold based on portfolio weights and bid,ask sizes
2) Connect to RabbitMQ for stream of orders


## References
<a id="1">[1]</a> 
https://stable-baselines.readthedocs.io/en/master/
<a id="2">[2]</a>
https://gym.openai.com/
