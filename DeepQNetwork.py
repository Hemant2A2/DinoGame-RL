import numpy as np
import random
from dino_world.envs.DinoGame import DinoGameEnv
import pygame
import torch
import torch.nn as nn
from collections import deque

# Define the DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layer_stack(x)
    

# Define the replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

# Define the agent
class Agent:
    def __init__(self, state_dim, action_dim, 
                 gamma=0.99, lr=1e-3, batch_size=64, 
                 epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, memory_capacity=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayMemory(memory_capacity)

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state, best_action=False):
        if random.random() > self.epsilon or best_action:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        states = torch.FloatTensor(batch[0])
        actions = torch.LongTensor(batch[1]).unsqueeze(1)
        rewards = torch.FloatTensor(batch[2])
        next_states = torch.FloatTensor(batch[3])
        dones = torch.FloatTensor(batch[4])

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Training loop
def learn(env, agent, num_episodes=100, target_update_freq=10):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update_policy()
            state = next_state
            total_reward += reward

        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()

# Main script
if __name__ == "__main__":
    env = DinoGameEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Agent(state_dim, action_dim,lr=0.01)
    learn(env, agent)