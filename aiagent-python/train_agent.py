import minerl
import gym
import random
import numpy as np

env = gym.make('MineRLNavigateDense-v0')

for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # random action
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode + 1} ended with reward: {total_reward}")

env.close()
