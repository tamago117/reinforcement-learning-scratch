import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10):
        self.rates = np.random.rand(k)

    def play(self, k):
        # 1: win, 0: lose
        if np.random.rand() < self.rates[k]:
            return 1
        else:
            return 0

class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.action_size = action_size
        self.q = np.zeros(action_size)
        self.n = np.zeros(action_size)
    
    def update(self, action, reward):
        self.n[action] += 1
        self.q[action] += (reward - self.q[action]) / self.n[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            # explore
            return np.random.randint(self.action_size)
        else:
            # exploit
            return np.argmax(self.q)

if __name__ == '__main__':
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        # 1 : get action
        action = agent.get_action()
        # 2 : play
        reward = bandit.play(action)
        # 3 : q update
        agent.update(action, reward)
        # 4 : log
        total_reward += reward
        total_rewards.append(total_reward)
        rates.append(total_reward/(step+1))
    
    plt.ylabel('total reward')
    plt.xlabel('steps')
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel('rate')
    plt.xlabel('steps')
    plt.plot(rates)
    plt.show()
    

