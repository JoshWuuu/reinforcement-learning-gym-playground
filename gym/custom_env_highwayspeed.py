import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random

class CarSpeedEnv(Env):
    def __init__(self):
        # Actions we can take, accelerate, decelerate, coast
        self.action_space = Discrete(3)
        # speed array
        self.observation_space = Box(low=np.array([0]), high=np.array([150]))
        # Set start temp
        self.state = 30 + random.randint(-3,3)
        # Set shower length
        self.road_length = 150
        
    def step(self, action):

        self.state += action -1 
        self.road_length -= 1 
        
        # Calculate reward
        if self.state >=65 and self.state <=75: 
            reward = 1 
        else: 
            reward = -1 
        
        # Check if shower is done
        if self.road_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = np.array([30 + random.randint(-3,3)])
        # Reset shower time
        self.shower_length = 150 
        return self.state

def main():
    env = CarSpeedEnv()
    # check_env(env, warn=True)
    env = DummyVecEnv([lambda: env])

    # test environment
    episodes = 5
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()

    # train model
    log_path = os.path.join('Training', 'Logs')
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=100000)

    # test model
    evaluate_policy(model, env, n_eval_episodes=10, render=False)

if __name__ == "__main__":
    main()