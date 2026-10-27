# %% [markdown]
# # Reinforcement Learning Lab — All 12 Simplified Codes
#
# **How to use this file in VS Code:** Each lab starts with `# %%`. Use **Run Cell**
# in the relevant lab section to execute just that experiment. Avoid **Run Python File**
# unless you intentionally want to run all 12 experiments sequentially; training labs
# may take a while.
#
# Install the common dependencies in your virtual environment:
#
# ```powershell
# pip install numpy torch gymnasium
# ```
#
# Labs 9 and 10 reflect the versions we discussed during preparation:
# - **Lab 9:** pure REINFORCE (policy only; no Critic), as specifically requested.
# - **Lab 10:** simplified target-floor Hybrid Actor–Critic, with the progress-based
#   reward and bounded speed used in the latest practice version. This differs from
#   the initially uploaded `lab10_Hybrid_Actor_critic.py`.
#
# The other lab sections preserve the uploaded simplified scripts, with formatting
# and section headers added for easier revision.
# %%
# ============================================================================
# LAB 1 — MULTI-ARMED BANDIT (MAB)
# ============================================================================
# NOTE: Compares five epsilon schedules for advertisement selection.
import numpy as np

probs = [0.3, 0.5, 0.67]  # Secret click probabilities for Ad 0, Ad 1, Ad 2
T = 500
variants = ["Standard", "Linear", "Exponential", "Step", "Inverse"]

for variant in variants:
    Q = np.zeros(3)       # Estimated value of each ad
    N = np.zeros(3)       # Number of times each ad was chosen
    regret = 0.0

    for t in range(1, T + 1):
        # 1. Calculate epsilon for the current step
        if variant == "Standard":
            eps = 0.2
        elif variant == "Linear":
            eps = max(0.05, 1.0 - (0.95 / T) * t)
        elif variant == "Exponential":
            eps = 0.05 + 0.95 * np.exp(-0.01 * t)
        elif variant == "Step":
            eps = max(0.05, 1.0 * (0.8 ** (t // 100)))
        elif variant == "Inverse":
            eps = 1.0 / (1 + 0.01 * t)

        # 2. Pick action: Explore (random) or Exploit (best Q)
        if np.random.rand() < eps:
            a = np.random.randint(3)
        else:
            a = np.argmax(Q)

        # 3. Get reward (1 if clicked, 0 if not)
        if np.random.rand() < probs[a]:
            r = 1
        else:
            r = 0

        # 4. Update count, Q-value, and cumulative regret
        N[a] += 1
        Q[a] += (1.0 / N[a]) * (r - Q[a])
        regret += max(probs) - probs[a]

    print(variant, "| Final Regret:", round(regret, 2), "| Q:", np.round(Q, 2))


# %%
# ============================================================================
# LAB 2 — VALUE ITERATION
# ============================================================================
import numpy as np

states = []
for r in range(10):
    for c in range(10):
        states.append((r, c))

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
names = ["Up", "Down", "Left", "Right"]

def step(s, a):
    nr = max(0, min(9, s[0] + actions[a][0]))
    nc = max(0, min(9, s[1] + actions[a][1]))
    done = (nr, nc) == (9, 9)
    r = 10.0 if done else -1.0
    return (nr, nc), r, done

def get_val(s, a, V, gamma):
    total = 0.0
    for actual_a in range(4):
        if actual_a == a:
            prob = 0.8
        else:
            prob = 0.2 / 3.0
        next_s, r, _ = step(s, actual_a)
        total += prob * (r + gamma * V[next_s])
    return total

def run_vi(gamma):
    V = {}
    policy = {}
    for s in states:
        V[s] = 0.0
        policy[s] = "GOAL"

    for _ in range(100):
        for s in states:
            if s == (9, 9):
                continue
            scores = []
            for a in range(4):
                scores.append(get_val(s, a, V, gamma))
            best_idx = np.argmax(scores)
            V[s] = scores[best_idx]
            policy[s] = names[best_idx]
    return V, policy

V_09, pol_09 = run_vi(0.9)
V_0, _ = run_vi(0.0)
V_1, _ = run_vi(1.0)

print("Gamma 0.9 -> V(0,0):", round(V_09[(0, 0)], 2), "| Best Action:", pol_09[(0, 0)])
print("Gamma 0.0 -> V(0,0):", round(V_0[(0, 0)], 2))
print("Gamma 1.0 -> V(0,0):", round(V_1[(0, 0)], 2))


# %%
# ============================================================================
# LAB 3 — POLICY ITERATION
# ============================================================================
import numpy as np

zones = ["A", "B", "C", "D", "E", "F", "G"]
neighbors = {
    "A": ["B"], "B": ["A", "C"], "C": ["B", "D"],
    "D": ["C", "E"], "E": ["D", "F"], "F": ["E", "G"], "G": ["F"]
}
demand = {"A": 0.8, "B": 0.4, "C": 0.5, "D": 0.1, "E": 0.3, "F": 0.6, "G": 0.9}

V = {}
policy = {}
for s in zones:
    V[s] = 0.0
    policy[s] = "Stay"
gamma = 0.9

def get_val(s, a):
    if a == "Stay":
        r = demand[s] * 20.0 + (1.0 - demand[s]) * (-5.0)
        return r + gamma * V[s]
    else:
        return -1.0 + gamma * (0.7 * V[a] + 0.3 * V[s])

for _ in range(20):
    # 1. Policy Evaluation
    for _ in range(50):
        for s in zones:
            V[s] = get_val(s, policy[s])

    # 2. Policy Improvement
    for s in zones:
        valid_actions = ["Stay"] + neighbors[s]
        scores = []
        for a in valid_actions:
            scores.append(get_val(s, a))
        best_idx = np.argmax(scores)
        policy[s] = valid_actions[best_idx]

for s in zones:
    print("Zone:", s, "| Best Action:", policy[s], "| Value:", round(V[s], 2))


# %%
# ============================================================================
# LAB 4 — MONTE CARLO PREDICTION
# ============================================================================
import numpy as np
from collections import defaultdict

def run_mc(step_fn, start_s, gamma, episodes):
    V = defaultdict(float)
    N = defaultdict(int)

    for _ in range(episodes):
        s = start_s
        ep_history = []
        done = False

        # 1. Play one episode
        while not done and len(ep_history) < 100:
            next_s, r, done = step_fn(s)
            ep_history.append((s, r))
            s = next_s

        # 2. Loop backward to compute return G
        G = 0.0
        visited = set()
        for s_step, r_step in reversed(ep_history):
            G = r_step + gamma * G
            if s_step not in visited:
                visited.add(s_step)
                N[s_step] += 1
                V[s_step] += (1.0 / N[s_step]) * (G - V[s_step])
    return V

# --- Part 1: 3-State MDP (Always Right) ---
def step_part1(s):
    if s == "S1":
        return "S2", 0.0, False
    elif s == "S2":
        return "S3", 1.0, True
    return "S3", 0.0, True

V1 = run_mc(step_part1, "S1", 0.9, 1000)
print("Part 1 Values:", dict(V1))

# --- Part 2: 4x4 Gridworld (Uniform Random Policy) ---
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def step_part2(s):
    a = np.random.randint(4)
    nr = s[0] + actions[a][0]
    nc = s[1] + actions[a][1]
    if not (0 <= nr < 4 and 0 <= nc < 4):
        return s, -1.0, False
    if (nr, nc) == (3, 3):
        return (3, 3), 10.0, True
    return (nr, nc), 0.0, False

V2 = run_mc(step_part2, (0, 0), 0.95, 10000)
print("\nPart 2 (4x4 Grid Values):")
for r in range(4):
    row_vals = []
    for c in range(4):
        row_vals.append(round(V2[(r, c)], 2))
    print(row_vals)


# %%
# ============================================================================
# LAB 5 — Q-LEARNING (DRONE NAVIGATION)
# ============================================================================
import numpy as np
from collections import defaultdict

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
obstacles = {(3,8), (4,8), (5,8), (10,2), (10,3), (10,4), (10,5)}
chargers = {(4,4), (11,11)}
goals = [(0, 15), (15, 0), (15, 15), (7, 12)]

def step(s, a):
    r, c, b, stage = s
    if 6 <= r <= 9 and 6 <= c <= 9 and np.random.rand() < 0.25:
        a = 1  # Wind drifts drone Down

    nr = r + actions[a][0]
    nc = c + actions[a][1]
    b -= 1
    reward = -1.0

    if not (0 <= nr < 16 and 0 <= nc < 16) or (nr, nc) in obstacles:
        nr, nc = r, c
        reward -= 10.0

    if (nr, nc) in chargers and b < 40:
        b = 40
        reward += 5.0

    if (nr, nc) == goals[stage]:
        if stage == 1:
            reward += 100.0
        elif stage == 3:
            reward += 300.0
        stage += 1

    done = (stage == 4) or (b <= 0)
    if b <= 0 and stage < 4:
        reward -= 200.0

    return (nr, nc, b, stage), reward, done

Q = defaultdict(lambda: np.zeros(4))
alpha = 0.1
gamma = 0.99

for ep in range(1, 5001):
    s = (0, 0, 40, 0)
    done = False
    total_r = 0.0
    eps = max(0.01, 1.0 * np.exp(-0.0005 * ep))

    while not done:
        if np.random.rand() < eps:
            a = np.random.randint(4)
        else:
            a = np.argmax(Q[s])

        next_s, r, done = step(s, a)
        Q[s][a] += alpha * (r + gamma * np.max(Q[next_s]) - Q[s][a])
        s = next_s
        total_r += r

    if ep % 1000 == 0:
        print("Episode:", ep, "| Total Reward:", total_r, "| Stage:", s[3])


# %%
# ============================================================================
# LAB 6 — SARSA (TRAFFIC QUEUES)
# ============================================================================
import numpy as np
from collections import defaultdict

def step(s, a):
    q_ns, q_ew = s
    arr_ns = np.random.poisson(1.2)
    arr_ew = np.random.poisson(1.0)

    clear_ns = 3 if a == 0 else 0
    clear_ew = 3 if a == 1 else 0

    next_ns = max(0, min(10, q_ns + arr_ns - clear_ns))
    next_ew = max(0, min(10, q_ew + arr_ew - clear_ew))
    reward = -(next_ns + next_ew)
    return (next_ns, next_ew), reward, False

Q = defaultdict(lambda: np.zeros(2))
alpha = 0.1
gamma = 0.95
eps = 0.1

for ep in range(1, 5001):
    s = (0, 0)
    if np.random.rand() < eps:
        a = np.random.randint(2)
    else:
        a = np.argmax(Q[s])

    total_wait = 0.0
    for _ in range(50):
        next_s, r, _ = step(s, a)
        if np.random.rand() < eps:
            next_a = np.random.randint(2)
        else:
            next_a = np.argmax(Q[next_s])

        # SARSA update uses actual next_a
        Q[s][a] += alpha * (r + gamma * Q[next_s][next_a] - Q[s][a])
        s = next_s
        a = next_a
        total_wait += -r

    if ep % 1000 == 0:
        print("Episode:", ep, "| Avg Waiting Cars:", round(total_wait / 50.0, 2))


# %%
# ============================================================================
# LAB 7 — DEEP Q-NETWORK (DQN)
# ============================================================================
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1")
net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
gamma = 0.99
eps = 0.2

for ep in range(1, 301):
    s, _ = env.reset()
    done = False
    total_r = 0.0

    while not done:
        s_t = torch.FloatTensor(s)
        if np.random.rand() < eps:
            a = env.action_space.sample()
        else:
            a = net(s_t).argmax().item()

        next_s, r, term, trunc, _ = env.step(a)
        done = term or trunc

        if done:
            next_q = 0.0
        else:
            next_q = net(torch.FloatTensor(next_s)).max().item()

        target = r + gamma * next_q
        loss = (net(s_t)[a] - target) ** 2

        opt.zero_grad()
        loss.backward()
        opt.step()

        s = next_s
        total_r += r

    eps = max(0.05, eps * 0.99)
    if ep % 50 == 0:
        print("DQN Episode:", ep, "| Score:", total_r)


# %%
# ============================================================================
# LAB 8 — TD-GAMMON-STYLE VALUE LEARNING
# ============================================================================
# NOTE: This is the simplified dice-move value-learning game supplied for practice, not a full backgammon implementation.
import torch
import torch.nn as nn
import numpy as np

net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
opt = torch.optim.SGD(net.parameters(), lr=0.05)

def get_moves(s, player):
    moves = []
    for dice in [1, 2, 3]:
        if player == 0:
            moves.append([min(10, s[0] + dice), s[1]])
        else:
            moves.append([s[0], min(10, s[1] + dice)])
    return moves

def pick_move(s, ply):
    moves = get_moves(s, player=0)
    if ply == 1:
        V = []
        for m in moves:
            score = net(torch.FloatTensor(m)).item()
            V.append(score)
        best_idx = np.argmax(V)
        return moves[best_idx]
    else:
        V = []
        for m in moves:
            opp_moves = get_moves(m, player=1)
            opp_scores = []
            for opp in opp_moves:
                score = net(torch.FloatTensor(opp)).item()
                opp_scores.append(score)
            worst_case = min(opp_scores)
            V.append(worst_case)
        best_idx = np.argmax(V)
        return moves[best_idx]

for ep in range(500):
    s = [0, 0]
    while s[0] < 10 and s[1] < 10:
        next_s = pick_move(s, ply=1)
        if next_s[1] < 10:
            opp_options = get_moves(next_s, player=1)
            next_s = opp_options[np.random.randint(3)]

        done = (next_s[0] >= 10) or (next_s[1] >= 10)
        if done:
            target = 1.0 if next_s[0] >= 10 else 0.0
        else:
            target = net(torch.FloatTensor(next_s)).item()

        loss = (net(torch.FloatTensor(s)) - target) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        s = next_s

print("1-Ply Move from [5, 5]:", pick_move([5, 5], ply=1))
print("2-Ply Move from [5, 5]:", pick_move([5, 5], ply=2))


# %%
# ============================================================================
# LAB 9 — PURE REINFORCE
# ============================================================================
# NOTE: Uses complete-episode discounted returns and an Actor only; no Critic.
import gymnasium as gym
import torch
import torch.nn as nn

env = gym.make("CartPole-v1")

# Policy network (Actor only — pure REINFORCE does not require a Critic)
policy = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
    nn.Softmax(dim=-1)
)

optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
gamma = 0.99

for ep in range(1, 501):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    total_reward = 0
    done = False

    # Collect a complete episode
    while not done:
        state_t = torch.FloatTensor(state)
        dist = torch.distributions.Categorical(policy(state_t))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        next_state, reward, term, trunc, _ = env.step(action.item())
        done = term or trunc
        rewards.append(reward)
        total_reward += reward
        state = next_state

    # Calculate discounted returns backwards
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

    # Policy-gradient loss
    loss = -torch.stack([
        log_prob * G for log_prob, G in zip(log_probs, returns)
    ]).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ep % 50 == 0:
        print("Pure REINFORCE Episode:", ep, "| Score:", total_reward)

env.close()


# %%
# ============================================================================
# LAB 10 — HYBRID ACTOR–CRITIC (ELEVATOR)
# ============================================================================
# NOTE: Latest simplified practice version discussed in chat: direction + bounded speed + Critic.
import torch
import torch.nn as nn

TARGET = 8.0


def step(s, direction, speed):
    speed = max(0.5, min(3.0, speed))
    next_floor = max(0.0, min(10.0, s[0] + (direction - 1) * speed))

    # Reward progress toward the target
    reward = abs(s[0] - TARGET) - abs(next_floor - TARGET) - 0.05
    done = abs(next_floor - TARGET) < 0.5

    if done:
        reward += 10.0

    return [next_floor], reward, done


# Direction Actor: Down, Stay, Up
actor_dir = nn.Sequential(
    nn.Linear(1, 16), nn.ReLU(),
    nn.Linear(16, 3), nn.Softmax(dim=-1)
)

# Speed Actor: continuous speed
actor_speed = nn.Sequential(
    nn.Linear(1, 16), nn.ReLU(),
    nn.Linear(16, 1), nn.Sigmoid()
)

# Critic: estimates state value
critic = nn.Sequential(
    nn.Linear(1, 16), nn.ReLU(),
    nn.Linear(16, 1)
)

optimizer = torch.optim.Adam(
    list(actor_dir.parameters())
    + list(actor_speed.parameters())
    + list(critic.parameters()),
    lr=0.001
)
gamma = 0.95

for ep in range(1, 1001):
    s = [0.0]
    total_reward = 0.0

    for _ in range(30):
        state_t = torch.tensor(s, dtype=torch.float32)

        direction_dist = torch.distributions.Categorical(actor_dir(state_t))
        speed_mean = 0.5 + 2.5 * actor_speed(state_t)
        speed_dist = torch.distributions.Normal(speed_mean, 0.5)

        direction = direction_dist.sample()
        speed = speed_dist.sample()

        next_s, reward, done = step(s, direction.item(), speed.item())

        value = critic(state_t)
        if done:
            next_value = 0.0
        else:
            next_value = critic(
                torch.tensor(next_s, dtype=torch.float32)
            ).item()

        td_target = reward + gamma * next_value
        advantage = td_target - value.item()

        actor_loss = -(
            direction_dist.log_prob(direction)
            + speed_dist.log_prob(speed)
        ) * advantage
        critic_loss = (td_target - value) ** 2
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        s = next_s
        total_reward += reward
        if done:
            break

    if ep % 100 == 0:
        print("Episode:", ep,
              "| Reward:", round(total_reward, 1),
              "| Floor:", round(s[0], 1))


# %%
# ============================================================================
# LAB 11 — SIMPLE VS. ADVANCED ACTOR–CRITIC
# ============================================================================
# NOTE: Compares using the TD target directly versus using the advantage for the Actor loss.
import torch
import torch.nn as nn
import random

TARGET = 8

def step(s, a):
    change = (a - 1) + random.randint(-1, 1)
    next_s = max(0, min(10, s[0] + change))
    fuel_cost = 2 if a == 2 else 0
    reward = -abs(next_s - TARGET) - fuel_cost
    done = (next_s == TARGET)
    return [next_s], reward, done

def run_ac(mode):
    actor = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 3), nn.Softmax(dim=-1))
    critic = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    gamma = 0.95

    for ep in range(200):
        s = [2.0]
        total_r = 0.0

        for _ in range(30):
            s_t = torch.FloatTensor(s)
            dist = torch.distributions.Categorical(actor(s_t))
            a = dist.sample()

            next_s, r, done = step(s, a.item())

            v_s = critic(s_t)
            if done:
                v_next = 0.0
            else:
                v_next = critic(torch.FloatTensor(next_s)).item()

            td_target = r + gamma * v_next
            advantage = td_target - v_s.item()

            if mode == "Simple":
                actor_loss = -dist.log_prob(a) * td_target
            else:
                actor_loss = -dist.log_prob(a) * advantage

            critic_loss = (td_target - v_s) ** 2
            loss = actor_loss + critic_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            s = next_s
            total_r += r
            if done:
                break
    return total_r

print("Simple Actor-Critic Final Reward  :", run_ac("Simple"))
print("Advanced Actor-Critic Final Reward:", run_ac("Advanced"))


# %%
# ============================================================================
# LAB 12 — PROXIMAL POLICY OPTIMIZATION (PPO)
# ============================================================================
# NOTE: Simplified PPO-style CartPole code from the uploaded file; uses a clipped policy objective.
import gymnasium as gym
import torch
import torch.nn as nn

env = gym.make("CartPole-v1")
actor = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2), nn.Softmax(dim=-1))
critic = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
gamma = 0.99

for ep in range(1, 301):
    s, _ = env.reset()
    done = False
    total_r = 0.0

    while not done:
        s_t = torch.FloatTensor(s)
        dist = torch.distributions.Categorical(actor(s_t))
        a = dist.sample()
        old_log_prob = dist.log_prob(a).item()

        next_s, r, term, trunc, _ = env.step(a.item())
        done = term or trunc

        if done:
            v_next = 0.0
        else:
            v_next = critic(torch.FloatTensor(next_s)).item()

        td_target = r + gamma * v_next
        advantage = td_target - critic(s_t).item()

        # PPO Clipped Update over 3 epochs
        for _ in range(3):
            new_dist = torch.distributions.Categorical(actor(s_t))
            ratio = torch.exp(new_dist.log_prob(a) - old_log_prob)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            actor_loss = -torch.min(surr1, surr2)

            v_s = critic(s_t)
            critic_loss = (td_target - v_s) ** 2
            loss = actor_loss + critic_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

        s = next_s
        total_r += r

    if ep % 50 == 0:
        print("PPO Episode:", ep, "| Score:", total_r)
