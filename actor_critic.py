if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from common.utils import plot_total_reward
#from torch.utils.tensorboard import SummaryWriter

class PolicyNet(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class ValueNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ActorCritic:
    def __init__(self, env, gamma=0.99, lr_pi=0.0002, lr_v=0.0005):
        self.env = env
        self.gamma = gamma

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.pi = PolicyNet(self.state_size, env.action_space.n)
        self.v = ValueNet(self.state_size)

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=lr_v)

        self.loss = nn.MSELoss()

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()

        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        # convert to tensor
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        next_state = torch.tensor(next_state[np.newaxis, :], dtype=torch.float32)

        # calculate TD target
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.detach()
        v = self.v(state)
        loss_v = self.loss(v, target)
        
        # update value network
        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.item()

        self.optimizer_pi.zero_grad()
        self.optimizer_v.zero_grad()
        loss_pi.backward()
        loss_v.backward()
        self.optimizer_pi.step()
        self.optimizer_v.step()

    def train(self, num_episodes=1000, render=False):
        total_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            #state = np.array(state)
            done = False
            total_reward = 0
            while not done:
                action, action_prob = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)[0:4]
                #next_state = np.array(next_state)
                self.update(state, action_prob, reward, next_state, done)
                state = next_state
                total_reward += reward
            total_rewards.append(total_reward)

            if episode % 100 == 0:
                print('Episode: {}, Total reward: {}'.format(episode, total_reward))
                #if render:
                    #self.env.render()

        return total_rewards

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    agent = ActorCritic(env)
    total_rewards = agent.train(num_episodes=1000, render=True)
    plot_total_reward(total_rewards)