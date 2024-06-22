
#this file is used to test the environment with the agent taking random actions

import gym
import pygame
import dino_world

env = gym.make('dino_world/DinoWorld-v0')
num_eps = 10
for ep in range(num_eps):
    obs = env.reset()
    env.render()
    done = False

    while not done:
        pygame.event.get()
        action = env.action_space.sample()  # Random action selection
        
        #action = 0
        print('Action:', action)    
        new_obs, reward, done, info= env.step(action)
        env.render()
        print('Reward:', reward)
        obs = new_obs
        pygame.time.wait(50)
        
env.close()