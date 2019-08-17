import numpy as np
import random
from datetime import datetime

random.seed(0)  # setting seed for generating random numbers


class qtable:
    def __init__(self, environment):
        self.env = environment
        print('Created Environment')
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n
        print('Action space size is :', self.action_size)
        print('State space size is :', self.state_size)

    def create_table(self, episodes=1000, steps=100, lr=0.5, gamma=0.75, min_epsilon=0.001, max_epsilon=1.0,
                     epsilon=1.0, decay_rate=0.01):
        start = datetime.now()
        self.q_table = np.zeros(shape=(self.state_size, self.action_size))
        print("You can access the qtable by using self.q_table.")
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            reward_total = 0
            done = False
            for step in range(steps):
                random_no = np.random.uniform(0, 1)
                if random_no <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state, :])
                new_state, reward, done, info = self.env.step(action)
                self.q_table[state, action] = self.q_table[state, action] + lr * (reward + gamma *
                np.max(self.q_table[new_state, :]) - self.q_table[state, action])
                reward_total += reward

                state = new_state
                if done:
                    break
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
            rewards.append(reward_total)
        print("Average Reward over Number of Episodes:", sum(rewards)/episodes)
        end = datetime.now()
        print("Total time taken to make qtable:", (end-start).total_seconds())

    def play_game(self, episodes=10, steps=100):
        start = datetime.now()
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            reward_total = 0
            print("*****")
            print("EPISODE:", episode)

            for step in range(steps):
                action = np.argmax(self.q_table[state, :])
                new_state, reward, done, info = self.env.step(action)
                reward_total+=reward
                state = new_state
                if done:
                    self.env.render()
                    print("Number of steps taken:", step+1)
                    print("State ended at:", state)
                    print("Total Reward received:", reward_total)
                    break
        self.env.close()
        end = datetime.now()
        print("Total time taken to play game:", (end-start).total_seconds())























































