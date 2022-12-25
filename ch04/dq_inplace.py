V = {"L1":0.0, "L2":0.0}
preV = V.copy()
action_L1 = {"left":-1.0, "right":1.0}
action_L2 = {"left":0.0, "right":-1.0}
action_prob = {"left":0.5, "right":0.5}
alpha = 0.8


cnt = 0
while True:
    V["L1"] = action_prob["left"] * (action_L1["left"] + alpha * V["L1"]) + action_prob["right"] * (action_L1["right"] + alpha * V["L2"])
    V["L2"] = action_prob["left"] * (action_L2["left"] + alpha * V["L1"]) + action_prob["right"] * (action_L2["right"] + alpha * V["L2"])

    delta = abs(V["L1"] - preV["L1"])
    delta = max(delta, abs(V["L2"] - preV["L2"]))
    print(delta)
    preV = V.copy()
    cnt += 1
    if delta < 0.0001:
        print(V)
        print(cnt)
        break