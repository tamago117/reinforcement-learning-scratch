import numpy as np
import matplotlib.pyplot as plt
from bandit import Agent

class NonStatBandit:
    def __init__(self, k=10):
        self.arms = k
        self.rates = np.random.rand(k)

    def play(self, k):
        rate = self.rates[k]
        self.rates += np.random.randn(self.arms) * 0.1 # add noise
        # 1: win, 0: lose
        if np.random.rand() < self.rates[k]:
            return 1
        else:
            return 0

class AlphaAgent:
    def __init__(self, epsilon, alpha, action_size=10):
        self.epsilon = epsilon
        self.alpha = alpha
        self.action_size = action_size
        self.q = np.zeros(action_size)

    def update(self, action, reward):
        self.q[action] += (reward - self.q[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            # explore
            return np.random.randint(self.action_size)
        else:
            # exploit
            return np.argmax(self.q)

if __name__ == '__main__':
    runs = 200
    steps = 1000
    epsilon = 0.1
    alpha = 0.8 # learning rate
    agent_type = ["sample average", "constant step size"]
    result = {}

    for agent in agent_type:
        all_rates = np.zeros((runs, steps))  # (200, 1000)

        for r in range(runs):
            bandit = NonStatBandit()
            if agent == "sample average":
                agent = Agent(epsilon)
            elif agent == "constant step size":
                agent = AlphaAgent(epsilon, alpha)

            total_reward = 0
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
                rates.append(total_reward/(step+1))
            all_rates[r] = rates
        avg_rates = np.average(all_rates, axis=0)
        result[agent] = avg_rates
    
    # plot
    plt.ylabel('average rates')
    plt.xlabel('steps')
    for key, avg_rate in result.items():
        plt.plot(avg_rate, label=key)
    plt.legend()
    plt.show()
