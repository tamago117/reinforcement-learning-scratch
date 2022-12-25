if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.gridworld import GridWorld
from collections import defaultdict

def eval_onestep(policy, V, env, gamma=0.9):
    # 全ての状態，行動について，状態価値関数を更新する
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0.0
            continue
    
        action_prob = policy[state]
        new_V = 0.0
        for action, action_prob in action_prob.items():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            new_V += action_prob * (reward + gamma * V[next_state])

        V[state] = new_V
    return V

def policy_eval(policy, V, env, gamma=1.0, threshold=1e-9):
    while True:
        old_V = V.copy()
        V = eval_onestep(policy, V, env, gamma)

        # 収束判定
        delta = 0.0
        for state in V.keys():
            delta = max(delta, abs(V[state] - old_V[state]))

        if delta < threshold:
            break
    return V

if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9

    policy = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0.0)

    V = policy_eval(policy, V, env, gamma)
    env.render_v(V)
    
