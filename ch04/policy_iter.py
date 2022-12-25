if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.gridworld import GridWorld
from collections import defaultdict
from ch04.policy_eval import policy_eval

def argmax(d):
    """d (dict)"""
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

# 価値関数を用いて方策を決定する
def greedy_policy(V, env, gamma=0.9):
    policy = {}

    for state in env.states():
        action_value = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            action_value[action] = reward + gamma * V[next_state]

        action_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        action_probs[argmax(action_value)] = 1.0
        policy[state] = action_probs

    return policy

def policy_iter(env, gamma=0.9, threshold=1e-9, is_render=True):
    policy = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0.0)

    while True:
        V = policy_eval(policy, V, env, gamma, threshold)
        new_policy = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, policy)

        # 収束判定
        if new_policy == policy:
            break
        policy = new_policy

    return policy, V

if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    policy, V = policy_iter(env, gamma, is_render=True)
