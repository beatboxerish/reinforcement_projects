import gym
from qtable import qtable

games = ["FrozenLake-v0", "Taxi-v2"]

env = gym.make(games[0])
obj = qtable(env)
obj.create_table(episodes=30000, lr=0.9, gamma=0.9)
obj.play_game(5)