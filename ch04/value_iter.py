if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_iter import greedy_policy

def value_iter_onestep(V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_value = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            action_value.append(reward + gamma * V[next_state])

        V[state] = max(action_value)

    return V

def value_iter(V, env, gamma=0.9, threshold=1e-3, is_render=True):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        diff = 0
        for state in env.states():
            diff = max(diff, abs(old_V[state] - V[state]))

        if diff < threshold:
            break

    return V

if __name__ == '__main__':
    env = GridWorld()
    V = defaultdict(lambda: 0.0)
    gamma = 0.9
    V = value_iter(V, env, gamma, is_render=True)

    policy = greedy_policy(V, env, gamma)
    env.render_v(V, policy)