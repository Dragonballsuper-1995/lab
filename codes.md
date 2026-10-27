## Reinforcement Learning Lab — Code Workbook

This workbook contains twelve self-contained reinforcement learning exercises. Each lab
includes its objective, required libraries, approximate run time, code, and representative
output.

## Contents

| Lab                                                    | Topic                                  | Libraries                                           | Approx. time |
| ------------------------------------------------------ | -------------------------------------- | --------------------------------------------------- | -----------: |
| [Lab 1](#lab-1--epsilon-greedy-advertisement-selection) | Epsilon-greedy advertisement selection | `numpy`, `matplotlib`                           |        5 sec |
| [Lab 2](#lab-2--state-value-function-estimation)        | State-value function estimation        | `numpy`, `pandas`                               |        5 sec |
| [Lab 3](#lab-3--policy-iteration)                       | Policy iteration                       | `numpy`                                           |        5 sec |
| [Lab 4](#lab-4--monte-carlo-prediction)                 | Monte Carlo prediction                 | `numpy`                                           |       10 sec |
| [Lab 5](#lab-5--q-learning)                             | Q-learning                             | `numpy`, `matplotlib`                           |       10 sec |
| [Lab 6](#lab-6--sarsa)                                  | SARSA                                  | `numpy`                                           |        5 sec |
| [Lab 7](#lab-7--deep-q-learning-for-cartpole)           | Deep Q-learning for CartPole           | `gymnasium`, `torch`, `numpy`, `matplotlib` |        1 min |
| [Lab 8](#lab-8--td-gammon-learning)                     | TD-Gammon learning                     | `torch`, `numpy`                                |      1.5 min |
| [Lab 9](#lab-9--reinforce-for-cartpole)                 | REINFORCE for CartPole                 | `gymnasium`, `torch`, `numpy`, `matplotlib` |   1.5–2 min |
| [Lab 10](#lab-10--actor-critic)                         | Actor-Critic                           | `torch`, `numpy`                                |       10 sec |
| [Lab 11](#lab-11--actor-critic-methods)                 | Actor-Critic methods                   | `torch`, `numpy`                                |       45 sec |
| [Lab 12](#lab-12--proximal-policy-optimization)         | Proximal Policy Optimization           | `gymnasium`, `torch`, `numpy`, `matplotlib` |       45 sec |

## Installation

Install the dependencies once before running the labs. The `%pip` commands below install
packages into the Python environment used by the current Jupyter kernel.

### Step 1 — Install the core libraries

```text
%pip install numpy pandas matplotlib gymnasium
```

### Step 2 — Install PyTorch

PyTorch is a larger download (approximately 120–250 MB). On Windows, the normal PyPI
package is CPU-only and is often faster to download than the build hosted on pytorch.org.

```text
%pip install torch --timeout 300 --retries 10
```

Alternatively:

```text
%pip install torch torchvision torchaudio
```

If the installation is slow or fails, try the CPU build from pytorch.org:

```text
%pip install torch --index-url https://download.pytorch.org/whl/cpu --timeout 300 --retries 10
```

If the download speed is only a few kB/s, try another network (for example, Wi-Fi or a
phone hotspot). If you get a permission error, add `--user`:

```text
%pip install --user numpy pandas matplotlib gymnasium torch
```

### Step 3 — Restart the kernel and verify the installation

In Jupyter, select **Kernel → Restart Kernel**, then run:

```python
import numpy, pandas, matplotlib, gymnasium, torch
print("numpy", numpy.__version__, "| pandas", pandas.__version__, "| matplotlib", matplotlib.__version__)
print("gymnasium", gymnasium.__version__, "| torch", torch.__version__)
```

The same commands work in Command Prompt or a terminal without the `%` prefix.

### Notes

- Use `gymnasium`, not the older `gym` package. The code expects
  `env.reset()` to return `(observation, info)` and `env.step()` to return five values.
  Any `gymnasium` version 0.26 or newer should work; CartPole needs no extra packages.
- The code was tested with Python 3.13, NumPy 2.3, pandas 3.0, Matplotlib 3.11,
  Gymnasium 1.3, and CPU-only PyTorch 2.14.
- Each code block was run in a fresh Jupyter kernel. Seeds are fixed, so results should be
  close to the representative output; small differences across versions are normal.
- Run times are approximate on a laptop CPU. Lab computers may be slower.

## Lab 1 — Epsilon-Greedy Advertisement Selection

**Objective**
  Implement different epsilon-greedy exploration strategies (fixed, linear, exponential,
  step and inverse-time decay) for advertisement selection using the Multi-Armed Bandit
  framework. Compare the strategies over 1000 trials (>= 500 required) using cumulative regret.

**Libraries:** `numpy`, `matplotlib`
**Approx. run time:** ~5 sec

```python
# --- 1. Setup: true click rates of 3 ads (unknown to the agent) ---
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1190)
Ads = [0.30, 0.50, 0.67]
best_ad = np.argmax(Ads)
total_trials = 1000


# --- 2. Epsilon schedules: fixed, linear, exponential, step and inverse-time decay ---
def get_epsilon(strat, step, eps_start=1, eps_end=0.05):
    if strat == "Standard":
        return 0.20
    elif strat == "Linear":
        eps = eps_start - ((eps_start - eps_end) / total_trials) * step
        return max(eps_end, eps)
    elif strat == "Exponential":
        return eps_end + (eps_start - eps_end) * np.exp(-0.005 * step)
    elif strat == "Step":
        eps = eps_start * (0.8 ** (step // 100))
        return max(eps_end, eps)
    elif strat == "Invtime":
        return eps_start / (1.0 + 0.01 * step)


# --- 3. Epsilon-greedy bandit: explore with prob eps, else show ad with best Q ---
def run_bandit(strat):
    regret_hist = []
    num_ads = len(Ads)
    Q = np.zeros(num_ads)
    N = np.zeros(num_ads)

    regret = 0.0
    for step in range(total_trials):
        eps = get_epsilon(strat, step)
        if np.random.rand() < eps:
            chosen_ad = np.random.randint(num_ads)
        else:
            chosen_ad = np.argmax(Q)

        reward = 1 if np.random.rand() < Ads[chosen_ad] else 0
        N[chosen_ad] += 1
        Q[chosen_ad] += (1.0 / N[chosen_ad]) * (reward - Q[chosen_ad])   # incremental mean

        regret += (Ads[best_ad] - Ads[chosen_ad])                       # clicks lost vs best ad
        regret_hist.append(regret)
    return regret_hist, Q


# --- 4. Compare strategies: lower cumulative regret = better ---
strategies = ["Standard", "Linear", "Exponential", "Step", "Invtime"]

for s in strategies:
    curve, Q = run_bandit(s)
    plt.plot(curve, label=s)
    print(f"{s:12s} final regret: {curve[-1]:6.2f} | estimated CTRs: {np.round(Q, 2)}")

plt.title("Regret over time for different epsilon strategies")
plt.xlabel("Trials")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.show()
```

**Representative output (approximate)**
  Standard     final regret:  47.60 | estimated CTRs: [0.29 0.46 0.66]
  Linear       final regret:  95.51 | estimated CTRs: [0.27 0.55 0.67]
  Exponential  final regret:  45.53 | estimated CTRs: [0.36 0.56 0.64]
  Step         final regret:  77.52 | estimated CTRs: [0.34 0.54 0.67]
  Invtime      final regret:  44.57 | estimated CTRs: [0.31 0.52 0.68]

+ regret plot (all curves flatten once the agent locks onto Ad C)

## Lab 2 — State-Value Function Estimation

**Objective**
  10x10 Grid World, start (0,0), goal (9,9). Actions Up/Down/Left/Right; intended move with
  prob 0.8, other directions with 0.2. Reward +10 at goal, -1 per move, gamma = 0.9.
  Represent the MDP and implement Value Iteration until convergence (threshold 1e-4).
  Print the optimal value function and optimal policy. Also compute for gamma = 0 and
  gamma = 1 and write observations. (Only numpy / pandas, no RL packages.)

**Libraries:** `numpy`, `pandas`
**Approx. run time:** ~5 sec

```python
# --- 1. MDP: 10x10 grid states, 4 actions, goal at (9,9) ---
import numpy as np
import pandas as pd
pd.set_option("display.width", 200)

grid_size = 10
goal_state = (9, 9)
actions = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
action_names = ["Up", "Down", "Left", "Right"]


def get_next_state(row, col, action):
    dr, dc = actions[action]
    next_row = max(0, min(grid_size - 1, row + dr))   # hitting a wall = stay in place
    next_col = max(0, min(grid_size - 1, col + dc))
    return (next_row, next_col)


# --- 2. Expected value of an action: 0.8 intended, 0.2 split over others; +10 goal, -1 per move ---
def get_action_value(row, col, intended_action, value_table, discount):
    expected_value = 0.0
    for act in action_names:
        prob = 0.8 if act == intended_action else 0.2 / 3.0
        next_row, next_col = get_next_state(row, col, act)
        reward = 10.0 if (next_row, next_col) == goal_state else -1
        expected_value += prob * (reward + discount * value_table[next_row, next_col])
    return expected_value


# --- 3. Value Iteration: V(s) = max_a sum P(s'|s,a)[R + gamma*V(s')] until change < 1e-4 ---
def run(discount=0.9, max_iter=1000, theta=1e-4):
    value_table = np.zeros((grid_size, grid_size))
    policy_table = np.full((grid_size, grid_size), "", dtype=object)

    for it in range(max_iter):
        max_change = 0.0
        new_value_table = np.copy(value_table)

        for row in range(grid_size):
            for col in range(grid_size):
                if (row, col) == goal_state:
                    policy_table[row, col] = "GOAL"
                    continue

                action_scores = [get_action_value(row, col, a, value_table, discount)
                                 for a in action_names]
                best_score = max(action_scores)
                new_value_table[row, col] = best_score
                policy_table[row, col] = action_names[np.argmax(action_scores)]
                max_change = max(max_change, abs(best_score - value_table[row, col]))

        value_table = new_value_table
        if max_change < theta:
            break

    print(f"(gamma = {discount}) converged after {it + 1} iterations")
    return value_table, policy_table


# --- 4. Run for gamma = 0.9, 0 and 1 ---
for g in [0.9, 0.0, 1.0]:
    values, policy = run(discount=g)
    print(f"--- OPTIMAL VALUE TABLE (Gamma = {g}) ---")
    print(pd.DataFrame(np.round(values, 2)))
    print(f"\n--- OPTIMAL POLICY (Gamma = {g}) ---")
    print(pd.DataFrame(policy))
    print("\n" + "=" * 70 + "\n")

print("OBSERVATIONS:")
print("gamma = 0  : only immediate reward matters -> almost all values are -1, policy is arbitrary")
print("             (ties -> 'Up') except cells next to the goal. Very short-sighted agent.")
print("gamma = 0.9: values rise smoothly towards the goal, policy is a clean Down/Right path.")
print("gamma = 1  : no discounting -> value ~ 10 - expected steps to goal, far cells very negative,")
print("             needs more iterations to converge but gives the same Down/Right policy.")
```

**Representative output (approximate)**
  (gamma = 0.9) converged after 41 iterations   -> V(0,0) = -8.04 ... V(8,9) = V(9,8) = 9.14
                                                   policy = Down / Right path to GOAL
  (gamma = 0.0) converged after 2 iterations    -> all -1.0 except 7.8 next to goal, policy mostly 'Up'
  (gamma = 1.0) converged after 52 iterations   -> V(0,0) = -12.82 ... 9.51 next to goal

+ full 10x10 value & policy tables for each gamma, then the OBSERVATIONS lines

## Lab 3 — Policy Iteration

**Objective**
  Taxi fleet management with Policy Iteration on a city of 7 zones (A-G). Actions: move to a
  neighbouring zone or stay. Rewards: pickup +20, moving -1, no passenger found -5.
  Move succeeds with prob 0.7, traffic keeps taxi in place with prob 0.3. Passenger demand
  varies by zone. Implement policy evaluation + policy improvement until the policy converges,
  print the optimal policy and value function, and compare with Value Iteration.

**Libraries:** `numpy`
**Approx. run time:** ~5 sec

```python
# --- 1. City map (7 zones in a line A-G) and passenger demand per zone ---
import numpy as np

zones = ["A", "B", "C", "D", "E", "F", "G"]
neighbors = {
    "A": ["B"], "B": ["A", "C"], "C": ["B", "D"], "D": ["C", "E"],
    "E": ["D", "F"], "F": ["E", "G"], "G": ["F"]
}
pickup_prob = {"A": 0.8, "B": 0.4, "C": 0.5, "D": 0.1, "E": 0.3, "F": 0.6, "G": 0.9}
discount_gamma = 0.9


# --- 2. Action value: Stay = pickup (+20) or no passenger (-5); Move = -1, 70% success / 30% traffic ---
def get_taxi_action_value(current_zone, action, value_table):
    if action == "Stay":
        p = pickup_prob[current_zone]
        expected_reward = (p * 20) + ((1 - p) * -5)
        return expected_reward + discount_gamma * value_table[current_zone]
    else:
        future_value = 0.7 * value_table[action] + 0.3 * value_table[current_zone]
        return -1.0 + discount_gamma * future_value


# --- 3. Policy Iteration: evaluate current policy, then improve greedily, until stable ---
policy = {zone: "Stay" for zone in zones}
value_table = {zone: 0.0 for zone in zones}
pi_rounds = 0

while True:
    pi_rounds += 1
    while True:                                   # policy evaluation
        max_change = 0.0
        for zone in zones:
            new_score = get_taxi_action_value(zone, policy[zone], value_table)
            max_change = max(max_change, abs(new_score - value_table[zone]))
            value_table[zone] = new_score
        if max_change < 1e-4:
            break

    policy_is_stable = True                       # policy improvement
    for zone in zones:
        old_action = policy[zone]
        valid_actions = ["Stay"] + neighbors[zone]
        action_scores = [get_taxi_action_value(zone, a, value_table) for a in valid_actions]
        policy[zone] = valid_actions[np.argmax(action_scores)]
        if policy[zone] != old_action:
            policy_is_stable = False
    if policy_is_stable:
        break

print(f"POLICY ITERATION converged in {pi_rounds} improvement rounds\n")
print("Zone | Optimal Action | Expected Long-Term Value")
print("-" * 48)
for zone in zones:
    label = "Stay & Pickup" if policy[zone] == "Stay" else f"Drive to {policy[zone]}"
    print(f"  {zone}  | {label:14s} | {value_table[zone]:.2f}")


# --- 4. Value Iteration on the same MDP for comparison ---
V_vi = {zone: 0.0 for zone in zones}
vi_iters = 0
while True:
    vi_iters += 1
    max_change = 0.0
    for zone in zones:
        best = max(get_taxi_action_value(zone, a, V_vi) for a in ["Stay"] + neighbors[zone])
        max_change = max(max_change, abs(best - V_vi[zone]))
        V_vi[zone] = best
    if max_change < 1e-4:
        break

policy_vi = {}
for zone in zones:
    acts = ["Stay"] + neighbors[zone]
    policy_vi[zone] = acts[np.argmax([get_taxi_action_value(zone, a, V_vi) for a in acts])]

print(f"\nVALUE ITERATION converged in {vi_iters} sweeps\n")
print("Zone | PI action | VI action | PI value | VI value")
print("-" * 52)
for zone in zones:
    print(f"  {zone}  | {policy[zone]:9s} | {policy_vi[zone]:9s} | {value_table[zone]:8.2f} | {V_vi[zone]:8.2f}")
print("\nSame policy from both:", policy == policy_vi)
print("Intuition: PI needs few (expensive) rounds, VI needs many cheap sweeps; both give the same")
print("optimal policy -> stay in high-demand zones (A, G), drive away from low-demand zones (D).")
```

**Representative output (approximate)**
  POLICY ITERATION converged in 3 improvement rounds
    A  | Stay & Pickup  | 150.00
    B  | Drive to A     | 128.08
    C  | Drive to B     | 109.17
    D  | Drive to E     | 108.91
    E  | Drive to F     | 127.79
    F  | Drive to G     | 149.66
    G  | Stay & Pickup  | 175.00
  VALUE ITERATION converged in 116 sweeps  -> same actions & values
  Same policy from both: True

## Lab 4 — Monte Carlo Prediction

**Objective**
  Part 1: 3-state MDP (S1, S2, S3), policy always Right, gamma = 0.9. Use First-Visit Monte
  Carlo prediction over >= 1000 episodes to estimate V(s).
  Part 2: 4x4 Gridworld (S0..S15), start S0, terminal S15, 0.8 intended / 0.2 random move,
  off-grid = -1, reaching S15 = +10, uniform random policy, gamma = 0.95. Estimate V(s) with

> = 10,000 episodes and print it as a 4x4 table.

**Libraries:** `numpy`
**Approx. run time:** ~10 sec

```python
# --- PART 1: 3-state MDP (S1, S2, S3), fixed policy = always Right, gamma = 0.9 ---
import numpy as np
np.random.seed(0)

states = ["S1", "S2", "S3"]
gamma = 0.9
V = {s: 0.0 for s in states}
returns = {s: [] for s in states}

def step_3state(s, a):
    if s == "S1":
        return ("S2", 0) if a == "Right" else ("S1", -1)
    elif s == "S2":
        return ("S3", 1) if a == "Right" else ("S1", 0)
    return ("S3", 0)

for ep in range(1000):
    s = str(np.random.choice(["S1", "S2"]))      # generate one episode
    episode = []
    while s != "S3":
        next_s, reward = step_3state(s, "Right")
        episode.append((s, reward))
        s = next_s

    # First-visit MC: walk backwards, G = r + gamma*G, V(s) = mean of returns
    G = 0
    visited = set()
    for i in range(len(episode) - 1, -1, -1):
        s_step, reward = episode[i]
        G = reward + gamma * G
        if s_step not in visited:
            visited.add(s_step)
            returns[s_step].append(G)
            V[s_step] = np.mean(returns[s_step])

print("Part 1 - 3-State Values:", {s: round(float(V[s]), 2) for s in states})
print("(Theory: V(S1)=0.9, V(S2)=1.0, V(S3)=0)")


# --- PART 2: 4x4 Gridworld, start S0, terminal S15, uniform random policy, gamma = 0.95 ---
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
gamma = 0.95
V_grid = np.zeros((4, 4))
returns_grid = {(r, c): [] for r in range(4) for c in range(4)}

def step_grid(r, c, chosen_idx):
    if np.random.rand() < 0.8:                    # 0.8 intended, 0.2 another direction
        a_idx = chosen_idx
    else:
        a_idx = np.random.choice([i for i in range(4) if i != chosen_idx])

    nr, nc = r + actions[a_idx][0], c + actions[a_idx][1]
    if not (0 <= nr < 4 and 0 <= nc < 4):
        return (r, c), -1
    if (nr, nc) == (3, 3):
        return (nr, nc), 10
    return (nr, nc), 0

for ep in range(10000):
    s = (0, 0)
    episode = []
    while s != (3, 3) and len(episode) < 100:   # cap length so random walk can't loop forever
        next_s, reward = step_grid(s[0], s[1], np.random.randint(4))
        episode.append((s, reward))
        s = next_s

    G = 0
    visited = set()
    for i in range(len(episode) - 1, -1, -1):
        s_step, reward = episode[i]
        G = reward + gamma * G
        if s_step not in visited:
            visited.add(s_step)
            returns_grid[s_step].append(G)

for (r, c), g_list in returns_grid.items():
    if g_list:
        V_grid[r, c] = np.mean(g_list)

print("\nPart 2 - 4x4 Gridworld Value Table (S0 top-left, S15 bottom-right):")
print(np.round(V_grid, 2))
```

**Representative output (approximate)**
  Part 1 - 3-State Values: {'S1': 0.9, 'S2': 1.0, 'S3': 0.0}
  Part 2 - 4x4 Gridworld Value Table:
  [[0.97 1.65 2.17 2.21]
   [1.62 2.56 3.51 3.87]
   [2.02 3.41 5.02 6.41]
   [2.02 3.68 6.35 0.]]

## Lab 5 — Q-Learning

**Objective**
  Custom 16x16 delivery-drone environment with battery (40 units), recharging stations (4,4)
  and (11,11), wind region (25% drift down), obstacles, and two sequential packages
  (P1: (0,15)->(15,0) +100, P2: (15,15)->(7,12) +300). Step -1, collision -10, recharge +5,
  out of battery -200. Train a tabular Q-learning agent: epsilon = max(0.01, e^(-0.00008 t)),
  gamma = 0.99, alpha = 0.1, 10,000 episodes, greedy evaluation (100 eps) every 500 episodes.
  NOTE: +5 recharge is paid once per station per episode, otherwise the agent learns to farm
  the charger forever instead of delivering. Distance-based reward shaping speeds up learning.

**Libraries:** `numpy`, `matplotlib`
**Approx. run time:** ~10 sec

```python
# --- 1. 16x16 drone environment: obstacles, chargers, wind, 2 sequential packages ---
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
np.random.seed(0)

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
obstacles = {(3, 8), (4, 8), (5, 8), (10, 2), (10, 3), (10, 4), (10, 5)}
chargers = {(4, 4), (11, 11)}
targets = [(0, 15), (15, 0), (15, 15), (7, 12)]     # P1 pickup, P1 drop, P2 pickup, P2 drop
MAX_STEPS = 400

# state = (row, col, battery, stage); stage = index of the next target, 4 = all delivered
def step_drone(state, a, charged):
    r, c, b, stage = state
    if 6 <= r <= 9 and 6 <= c <= 9 and np.random.rand() < 0.25:     # wind drifts drone down
        nr, nc = r + 1, c
    else:
        nr, nc = r + actions[a][0], c + actions[a][1]

    b -= 1
    reward = -1
    if not (0 <= nr < 16 and 0 <= nc < 16) or (nr, nc) in obstacles:
        nr, nc = r, c
        reward -= 10
    if (nr, nc) in chargers and b < 40:
        b = 40
        if (nr, nc) not in charged:          # +5 paid once per station (stops charger farming)
            charged.add((nr, nc))
            reward += 5

    if (nr, nc) == targets[stage]:
        stage += 1
        reward += {2: 100, 4: 300}.get(stage, 0)

    if stage == 4: return (nr, nc, b, stage), reward, True
    if b <= 0:     return (nr, nc, 0, stage), reward - 200, True
    return (nr, nc, b, stage), reward, False


# --- 2. Potential = -(distance left to finish both deliveries); used only to speed up learning ---
dist = lambda p, q: abs(p[0] - q[0]) + abs(p[1] - q[1])
legs_left = [sum(dist(targets[i], targets[i + 1]) for i in range(k, 3)) for k in range(4)] + [0]

def potential(s):
    r, c, b, stage = s
    return 0.0 if stage == 4 else -float(dist((r, c), targets[stage]) + legs_left[stage])


# --- 3. Tabular Q-learning, epsilon-greedy with exponential decay ---
Q = defaultdict(lambda: np.zeros(4))
alpha, gamma = 0.1, 0.99
t = 0

def run_episode(train=True):
    global t
    s, charged = (0, 0, 40, 0), set()
    total_reward, done, steps = 0, False, 0
    while not done and steps < MAX_STEPS:
        eps = max(0.01, 1.0 * np.exp(-0.00008 * t)) if train else 0.0
        a = np.random.randint(4) if np.random.rand() < eps else int(np.argmax(Q[s]))
        next_s, reward, done = step_drone(s, a, charged)
        if train:
            shaped = reward + gamma * potential(next_s) - potential(s)
            target = shaped + (0 if done else gamma * np.max(Q[next_s]))
            Q[s][a] += alpha * (target - Q[s][a])
            t += 1
        s = next_s
        total_reward += reward
        steps += 1
    return total_reward, (s[3] == 4)


# --- 4. Train 10,000 episodes; every 500 run 100 greedy (eps=0) evaluation episodes ---
eval_eps, eval_rewards, eval_success = [], [], []
for ep in range(1, 10001):
    run_episode(train=True)
    if ep % 500 == 0:
        results = [run_episode(train=False) for _ in range(100)]
        avg_rew = np.mean([res[0] for res in results])
        win_rate = np.mean([res[1] for res in results]) * 100
        eval_eps.append(ep); eval_rewards.append(avg_rew); eval_success.append(win_rate)
        print(f"Episode {ep:5d} | Greedy Avg Reward: {avg_rew:7.1f} | Completion: {win_rate:5.1f}%")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(eval_eps, eval_rewards, marker="o"); ax[0].set_title("Greedy mean reward"); ax[0].set_xlabel("Episode")
ax[1].plot(eval_eps, eval_success, marker="o"); ax[1].set_title("Completion rate (%)"); ax[1].set_xlabel("Episode")
plt.tight_layout(); plt.show()
```

**Representative output (approximate)**
  Episode   500 | Greedy Avg Reward:  -240.0 | Completion:   0.0%
  Episode  1000 | Greedy Avg Reward:  -240.0 | Completion:   0.0%
  Episode  1500 | Greedy Avg Reward:   322.0 | Completion: 100.0%
  ...  (stays at 322.0 / 100.0% up to episode 10000)

+ 2 plots (mean reward, completion rate)

## Lab 6 — SARSA

**Objective**
  Traffic-light control at a 4-way intersection. Cars arrive with a Poisson distribution;
  the agent keeps or switches the green light to minimise waiting time / queue length.
  Implement SARSA with epsilon-greedy and a Q-table, train over many episodes (simulated
  days), compare against a fixed-cycle baseline, and compare 5000 vs 10000 training episodes.

**Libraries:** `numpy`
**Approx. run time:** ~5 sec

```python
# --- 1. Traffic environment: state = (NS queue, EW queue, green light 0=NS / 1=EW) ---
import numpy as np
from collections import defaultdict
np.random.seed(0)

def step_traffic(s, a):
    ns_q, ew_q, light = s
    new_light = light if a == 0 else 1 - light          # action 0 = keep, 1 = switch

    ns_q += np.random.poisson(1.5)                      # Poisson car arrivals
    ew_q += np.random.poisson(1.0)

    clear_cap = 3 if a == 0 else 2                      # switching wastes some green time
    if new_light == 0:
        ns_q = max(0, ns_q - clear_cap)
    else:
        ew_q = max(0, ew_q - clear_cap)

    ns_q, ew_q = min(10, ns_q), min(10, ew_q)
    return (ns_q, ew_q, new_light), -(ns_q + ew_q)      # reward = - waiting cars


# --- 2. SARSA with epsilon-greedy: update uses the ACTUAL next action (on-policy) ---
def train_sarsa(num_episodes):
    Q = defaultdict(lambda: np.zeros(2))
    alpha, gamma, eps = 0.1, 0.95, 0.1

    def choose(s):
        return np.random.randint(2) if np.random.rand() < eps else int(np.argmax(Q[s]))

    for ep in range(num_episodes):
        s = (0, 0, 0)
        a = choose(s)
        for step in range(50):
            next_s, reward = step_traffic(s, a)
            next_a = choose(next_s)
            Q[s][a] += alpha * (reward + gamma * Q[next_s][next_a] - Q[s][a])
            s, a = next_s, next_a
    return Q


# --- 3. Compare fixed-cycle baseline (switch every 5 steps) vs SARSA at 5000 / 10000 episodes ---
def evaluate_policy(Q=None):
    total_wait = 0
    for ep in range(100):
        s = (0, 0, 0)
        for step in range(50):
            a = (1 if step % 5 == 0 else 0) if Q is None else int(np.argmax(Q[s]))
            s, reward = step_traffic(s, a)
            total_wait += -reward
    return total_wait / (100 * 50)

print(f"Fixed-Cycle Baseline Avg Queue  : {evaluate_policy(None):.2f} cars")
print(f"SARSA (5,000 Episodes) Avg Queue : {evaluate_policy(train_sarsa(5000)):.2f} cars")
print(f"SARSA (10,000 Episodes) Avg Queue: {evaluate_policy(train_sarsa(10000)):.2f} cars")
```

**Representative output (approximate)**
  Fixed-Cycle Baseline Avg Queue  : 7.47 cars
  SARSA (5,000 Episodes) Avg Queue : 4.96 cars
  SARSA (10,000 Episodes) Avg Queue: 4.65 cars

## Lab 7 — Deep Q-Learning for CartPole

**Objective**
  Implement a Python program using Deep Q-Learning (DQN) to learn the CartPole-v1 task
  (Q-network, experience replay, target network, epsilon-greedy).

**Libraries:** `gymnasium`, `torch`, `numpy`, `matplotlib`
**Approx. run time:** ~1 min

```python
# --- 1. Imports & seeds ---
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

random.seed(0); np.random.seed(0); torch.manual_seed(0)


# --- 2. DQN: Q-network + target network + replay memory ---
def train_dqn(episodes=250):
    env = gym.make("CartPole-v1")
    q_net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    target_net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    memory = deque(maxlen=10000)

    eps, gamma, steps = 1.0, 0.99, 0
    scores = []

    for ep in range(episodes):
        s, _ = env.reset(seed=ep)
        ep_reward, done = 0, False

        while not done:
            if random.random() < eps:                       # epsilon-greedy action
                a = random.randint(0, 1)
            else:
                with torch.no_grad():
                    a = q_net(torch.FloatTensor(s)).argmax().item()

            next_s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            memory.append((s, a, r, next_s, float(term)))
            s = next_s
            ep_reward += r
            steps += 1

            # --- 3. Train on a random mini-batch: target = r + gamma * max_a' Q_target(s', a') ---
            if len(memory) >= 64:
                batch = random.sample(memory, 64)
                states = torch.FloatTensor(np.array([b[0] for b in batch]))
                acts = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
                rewards = torch.FloatTensor([b[2] for b in batch])
                next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
                dones = torch.FloatTensor([b[4] for b in batch])

                current_q = q_net(states).gather(1, acts).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                target_q = rewards + gamma * next_q * (1 - dones)

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps % 500 == 0:                            # copy weights to target network
                target_net.load_state_dict(q_net.state_dict())

        eps = max(0.05, eps * 0.97)
        scores.append(ep_reward)
        if (ep + 1) % 25 == 0:
            print(f"DQN Episode {ep+1:3d} | eps {eps:.2f} | Avg Score (last 25): {np.mean(scores[-25:]):.1f}")
    env.close()
    return scores


# --- 4. Train and plot (CartPole max score = 500) ---
dqn_scores = train_dqn(250)
plt.plot(dqn_scores); plt.xlabel("Episode"); plt.ylabel("Score"); plt.title("DQN on CartPole-v1"); plt.show()
```

**Representative output (approximate)**
  DQN Episode  25 | eps 0.47 | Avg Score (last 25): 14.5
  DQN Episode 100 | eps 0.05 | Avg Score (last 25): 22.8
  DQN Episode 150 | eps 0.05 | Avg Score (last 25): 221.8
  DQN Episode 200 | eps 0.05 | Avg Score (last 25): 140.0
  DQN Episode 250 | eps 0.05 | Avg Score (last 25): 212.1      (max possible = 500)

+ score plot   (DQN scores go up and down a lot - that is normal for DQN)

## Lab 8 — TD-Gammon Learning

**Objective**
  Implement a program to learn a TD-Gammon style game using Temporal Difference learning
  (neural network evaluates positions, self-play training updates the weights). Train with
  different hyper-parameters and observe the difference between 1-ply search (one agent move)
  and 2-ply search (agent move + opponent reply) in move selection.
  (A simplified backgammon race is used: 2 dice of 1-3, first to square 12 wins, landing on
  the opponent sends it back to start.)

**Libraries:** `torch`, `numpy`
**Approx. run time:** ~1.5 min

```python
# --- 1. Mini backgammon-style race: reach square 12 first; landing on opponent sends it home ---
import torch
import torch.nn as nn
import numpy as np
import random
random.seed(0); np.random.seed(0); torch.manual_seed(0)

BOARD = 12
ALL_ROLLS = [(d1, d2) for d1 in [1, 2, 3] for d2 in [1, 2, 3]]

def get_next_states(state, turn, dice):
    d1, d2 = dice
    p1, p2 = state
    result = []
    for m in sorted({d1, d2, d1 + d2}):                     # move by die 1, die 2 or both
        if turn == 1:
            n1, n2 = min(BOARD, p1 + m), p2
            if n1 == n2 and n1 < BOARD: n2 = 0
        else:
            n1, n2 = p1, min(BOARD, p2 + m)
            if n2 == n1 and n2 < BOARD: n1 = 0
        result.append([n1, n2])
    return result

def winner(s):
    return 1 if s[0] >= BOARD else (2 if s[1] >= BOARD else 0)


# --- 2. TD network: V(s) = probability that Player 1 wins ---
def create_td_net(hidden=16):
    return nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())

def features(s, turn):
    return torch.FloatTensor([s[0] / BOARD, s[1] / BOARD, 1.0 if turn == 1 else 0.0])

def value(net, s, turn):
    w = winner(s)
    if w: return 1.0 if w == 1 else 0.0
    with torch.no_grad():
        return net(features(s, turn)).item()


# --- 3. Move selection: 1-ply = best V after my move; 2-ply = also average opponent's best reply ---
def select_move(net, state, turn, dice, ply=1):
    opp = 3 - turn
    candidates = get_next_states(state, turn, dice)
    scores = []
    for c in candidates:
        if ply == 1 or winner(c):
            scores.append(value(net, c, opp))
        else:
            replies = []
            for d in ALL_ROLLS:
                rv = [value(net, r, turn) for r in get_next_states(c, opp, d)]
                replies.append(max(rv) if opp == 1 else min(rv))
            scores.append(np.mean(replies))
    best = np.argmax(scores) if turn == 1 else np.argmin(scores)
    return candidates[best]


# --- 4. TD(0) self-play training: push V(s) towards V(s') (or 1/0 at game end) ---
def train_td_gammon(lr=0.05, gamma=1.0, hidden=16, episodes=1500, explore=0.1):
    net = create_td_net(hidden)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for ep in range(episodes):
        s, turn = [0, 0], 1
        while not winner(s):
            dice = (random.randint(1, 3), random.randint(1, 3))
            if random.random() < explore:
                next_s = random.choice(get_next_states(s, turn, dice))
            else:
                next_s = select_move(net, s, turn, dice, ply=1)

            w = winner(next_s)
            if w:
                target = torch.tensor([1.0 if w == 1 else 0.0])
            else:
                with torch.no_grad():
                    target = gamma * net(features(next_s, 3 - turn))
            loss = (net(features(s, turn)) - target).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            s, turn = next_s, 3 - turn
    return net


def win_rate(player_a, player_b, games=200):
    wins = 0
    for g in range(games):
        a_first = (g % 2 == 0)                               # swap who starts every game
        s, turn = [0, 0], 1
        while not winner(s):
            dice = (random.randint(1, 3), random.randint(1, 3))
            mover = player_a if (turn == 1) == a_first else player_b
            s = mover(s, turn, dice)
            turn = 3 - turn
        wins += (winner(s) == 1) == a_first
    return 100.0 * wins / games

random_player = lambda s, turn, dice: random.choice(get_next_states(s, turn, dice))


# --- 5. For each hyper-parameter setting: 1-ply and 2-ply vs random, then 2-ply vs 1-ply ---
configs = [dict(lr=0.05, gamma=1.0, hidden=16, episodes=1500),
           dict(lr=0.005, gamma=1.0, hidden=16, episodes=1500),
           dict(lr=0.05, gamma=0.9, hidden=32, episodes=1500),
           dict(lr=0.05, gamma=1.0, hidden=16, episodes=200)]

print("lr     gamma hidden episodes | 1-ply vs random | 2-ply vs random | 2-ply vs 1-ply")
for cfg in configs:
    net = train_td_gammon(**cfg)
    one_ply = lambda s, t, d, n=net: select_move(n, s, t, d, ply=1)
    two_ply = lambda s, t, d, n=net: select_move(n, s, t, d, ply=2)
    print(f"{cfg['lr']:<6} {cfg['gamma']:<5} {cfg['hidden']:<6} {cfg['episodes']:<8} | "
          f"{win_rate(one_ply, random_player, 400):14.1f}% | {win_rate(two_ply, random_player, 400):14.1f}% | "
          f"{win_rate(two_ply, one_ply, 400):13.1f}%")

print()
print("2-ply looks one opponent move ahead (averaged over all dice rolls), so it avoids squares where")
print("it can be hit -> it wins >= 50% against 1-ply (it is never worse) but costs ~27x more evaluations.")
```

**Representative output (approximate)**
  lr     gamma hidden episodes | 1-ply vs random | 2-ply vs random | 2-ply vs 1-ply
  0.05   1.0   16     1500     |           87.8% |           88.2% |          57.5%
  0.005  1.0   16     1500     |           81.2% |           87.2% |          64.5%
  0.05   0.9   32     1500     |           86.2% |           91.2% |          50.5%
  0.05   1.0   16     200      |           88.0% |           85.5% |          52.5%

## Lab 9 — REINFORCE for CartPole

**Objective**
  Implement the REINFORCE (Monte-Carlo policy gradient) algorithm to train CartPole and
  evaluate its performance against a Deep Q-Learning baseline.
  (The DQN baseline is included, so this block runs on its own without Lab 7.)

**Libraries:** `gymnasium`, `torch`, `numpy`, `matplotlib`
**Approx. run time:** ~1.5–2 min

```python
# --- 1. Imports & seeds ---
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

random.seed(0); np.random.seed(0); torch.manual_seed(0)


# --- 2. REINFORCE: play a full episode, then increase log-prob of actions weighted by return G ---
def train_reinforce(episodes=250):
    env = gym.make("CartPole-v1")
    policy_net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(dim=-1))
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-2)
    gamma = 0.99
    scores = []

    for ep in range(episodes):
        s, _ = env.reset(seed=ep)
        log_probs, rewards = [], []
        done = False
        while not done:
            dist = torch.distributions.Categorical(policy_net(torch.FloatTensor(s)))
            a = dist.sample()
            s, r, term, trunc, _ = env.step(a.item())
            done = term or trunc
            log_probs.append(dist.log_prob(a))
            rewards.append(r)
        scores.append(sum(rewards))

        G, returns = 0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # normalise for stability

        loss = -(torch.stack(log_probs) * returns).sum()                # policy gradient loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 25 == 0:
            print(f"REINFORCE Episode {ep+1:3d} | Avg Score (last 25): {np.mean(scores[-25:]):.1f}")
    env.close()
    return scores


# --- 3. DQN baseline (same as Lab 7) so this cell runs on its own ---
def train_dqn(episodes=250):
    env = gym.make("CartPole-v1")
    q_net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    target_net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    memory = deque(maxlen=10000)
    eps, gamma, steps, scores = 1.0, 0.99, 0, []

    for ep in range(episodes):
        s, _ = env.reset(seed=ep)
        ep_reward, done = 0, False
        while not done:
            if random.random() < eps:
                a = random.randint(0, 1)
            else:
                with torch.no_grad():
                    a = q_net(torch.FloatTensor(s)).argmax().item()
            next_s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            memory.append((s, a, r, next_s, float(term)))
            s = next_s
            ep_reward += r
            steps += 1

            if len(memory) >= 64:
                batch = random.sample(memory, 64)
                states = torch.FloatTensor(np.array([b[0] for b in batch]))
                acts = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
                rewards = torch.FloatTensor([b[2] for b in batch])
                next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
                dones = torch.FloatTensor([b[4] for b in batch])
                current_q = q_net(states).gather(1, acts).squeeze(1)
                with torch.no_grad():
                    target_q = rewards + gamma * target_net(next_states).max(1)[0] * (1 - dones)
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if steps % 500 == 0:
                target_net.load_state_dict(q_net.state_dict())

        eps = max(0.05, eps * 0.97)
        scores.append(ep_reward)
        if (ep + 1) % 25 == 0:
            print(f"DQN Episode {ep+1:3d} | Avg Score (last 25): {np.mean(scores[-25:]):.1f}")
    env.close()
    return scores


# --- 4. Train both for the same number of episodes and compare ---
reinforce_scores = train_reinforce(400)
dqn_scores = train_dqn(400)

print(f"\nREINFORCE avg of last 50 episodes: {np.mean(reinforce_scores[-50:]):.1f}")
print(f"DQN       avg of last 50 episodes: {np.mean(dqn_scores[-50:]):.1f}")
print("REINFORCE: simple, learns the policy directly but noisy (high-variance Monte-Carlo returns).")
print("DQN: replay memory + target network reuse data -> usually learns faster, but can be unstable.")

plt.plot(reinforce_scores, label="REINFORCE")
plt.plot(dqn_scores, label="DQN Baseline")
plt.xlabel("Episode"); plt.ylabel("Score")
plt.legend(); plt.title("REINFORCE vs DQN on CartPole"); plt.show()
```

**Representative output (approximate)**
  REINFORCE Episode 100 | Avg Score (last 25): 23.9
  REINFORCE Episode 400 | Avg Score (last 25): 121.4
  DQN Episode 100 | Avg Score (last 25): 16.7
  DQN Episode 400 | Avg Score (last 25): 206.6
  REINFORCE avg of last 50 episodes: 146.9
  DQN       avg of last 50 episodes: 196.1

+ comparison plot

## Lab 10 — Actor-Critic

**Objective**
  Train an elevator-control agent with the Actor-Critic algorithm. The agent learns a discrete
  action (down / stay / up) AND a continuous action (elevator speed), with the objective of
  minimising passenger waiting time.

**Libraries:** `torch`, `numpy`
**Approx. run time:** ~10 sec

```Python
# --- 1. Hybrid Actor-Critic: discrete head (Down/Stay/Up) + continuous speed head + critic V(s) ---
import torch
import torch.nn as nn
import numpy as np
np.random.seed(0); torch.manual_seed(0)

class HybridActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.discrete_head = nn.Sequential(nn.Linear(32, 3), nn.Softmax(dim=-1))
        self.speed_mu = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.critic = nn.Linear(32, 1)

    def forward(self, s):
        x = self.shared(s)
        mu = self.speed_mu(x) * 2.0 + 1.0            # mean speed scaled to [1, 3] floors/step
        return self.discrete_head(x), mu, self.critic(x)

def get_state(elev, passenger):                  # [elevator floor, passenger floor, distance]
    return torch.FloatTensor([elev / 9, passenger / 9, (passenger - elev) / 9])

model = HybridActorCritic()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
gamma = 0.95


# --- 2. One episode: elevator (floors 0-9) must reach the waiting passenger quickly ---
def run_episode(train=True):
    elev = float(np.random.randint(0, 10))
    passenger = float(np.random.choice([f for f in range(10) if f != elev]))
    total_wait = 0
    for step in range(20):
        probs, mu, v_s = model(get_state(elev, passenger))
        dist_dir = torch.distributions.Categorical(probs)
        dist_spd = torch.distributions.Normal(mu, 0.3)
        if train:
            a_dir, a_spd = dist_dir.sample(), torch.clamp(dist_spd.sample(), 1.0, 3.0)
        else:
            a_dir, a_spd = probs.argmax(), mu.detach().clamp(1.0, 3.0)   # greedy test

        elev = float(np.clip(elev + (a_dir.item() - 1) * a_spd.item(), 0.0, 9.0))
        gap = abs(elev - passenger)
        done = gap < 0.8
        reward = (10.0 if done else -gap - 0.2 * a_spd.item()) / 10   # wait + energy penalty
        total_wait += gap

        # --- 3. Actor-Critic update: advantage = r + gamma*V(s') - V(s) ---
        if train:
            _, _, v_next = model(get_state(elev, passenger))
            advantage = reward + (0.0 if done else gamma * v_next.detach()) - v_s
            actor_loss = -(dist_dir.log_prob(a_dir) + dist_spd.log_prob(a_spd)) * advantage.detach()
            loss = (actor_loss + advantage.pow(2)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if done: break
    return total_wait, step + 1, done


# --- 4. Train 600 episodes, then test the greedy policy ---
waits, steps = [], []
for ep in range(1, 601):
    w, s, _ = run_episode(train=True)
    waits.append(w); steps.append(s)
    if ep % 100 == 0:
        print(f"Episode {ep:3d} | Avg wait metric (last 100): {np.mean(waits[-100:]):5.1f} "
              f"| Avg steps to reach passenger: {np.mean(steps[-100:]):4.1f}")

test = [run_episode(train=False) for _ in range(200)]
print(f"\nGreedy test (200 calls): avg steps {np.mean([t[1] for t in test]):.2f} | "
      f"reached passenger {100 * np.mean([t[2] for t in test]):.0f}% | avg wait {np.mean([t[0] for t in test]):.2f}")
```

**Representative output (approximate)**
  Episode 100 | Avg wait metric (last 100):  18.1 | Avg steps to reach passenger:  5.5
  Episode 300 | Avg wait metric (last 100):   4.9 | Avg steps to reach passenger:  2.7
  Episode 600 | Avg wait metric (last 100):   3.9 | Avg steps to reach passenger:  2.7
  Greedy test (200 calls): avg steps 3.37 | reached passenger 96% | avg wait 4.53

## Lab 11 — Actor-Critic Methods

**Objective**
  Design an actor-critic agent to optimise the fuel efficiency of an autonomous delivery
  vehicle by deciding when to accelerate, brake or maintain speed under varying traffic /
  road speed limits. Compare Simple Actor-Critic with Advanced (Advantage) Actor-Critic.

**Libraries:** `torch`, `numpy`
**Approx. run time:** ~45 sec

```python
# --- 1. Delivery vehicle environment: state = [current speed, road speed limit] ---
import torch
import torch.nn as nn
import numpy as np
np.random.seed(0); torch.manual_seed(0)

def step_vehicle(s, a):
    speed, target_speed = s
    accel = (a - 1) * 2.0                            # 0 = brake (-2), 1 = maintain, 2 = accelerate (+2)
    next_speed = float(np.clip(speed + accel + np.random.uniform(-0.5, 0.5), 0, 20))
    fuel_cost = 1.5 if a == 2 else (0.2 if a == 1 else 0.5)
    reward = 10.0 - abs(next_speed - target_speed) - fuel_cost
    return [next_speed, target_speed], reward


# --- 2. Actor-Critic training: actor = policy pi(a|s), critic = V(s) ---
def train_vehicle_ac(use_advantage=True, episodes=400):
    actor = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 3), nn.Softmax(dim=-1))
    critic = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-3)
    gamma = 0.95
    history = []

    for ep in range(episodes):
        s = [5.0, float(np.random.choice([8.0, 12.0, 16.0]))]    # different road / traffic limits
        ep_reward = 0
        for step in range(30):
            s_tensor = torch.FloatTensor(s) / 20.0
            dist = torch.distributions.Categorical(actor(s_tensor))
            a = dist.sample()
            v_s = critic(s_tensor)
            next_s, r = step_vehicle(s, a.item())
            ep_reward += r

            v_next = critic(torch.FloatTensor(next_s) / 20.0).detach()
            td_target = r + gamma * v_next
            td_error = td_target - v_s

            # Simple AC weights log-prob by Q estimate (td_target); Advanced uses Advantage (td_error)
            weight = td_error.detach() if use_advantage else td_target.detach()
            loss = (-dist.log_prob(a) * weight + td_error.pow(2)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            s = next_s
        history.append(ep_reward)
    return history


# --- 3. Compare Simple vs Advanced (Advantage) Actor-Critic ---
simple = train_vehicle_ac(use_advantage=False)
advanced = train_vehicle_ac(use_advantage=True)
print(f"Simple Actor-Critic   avg reward (last 50 eps): {np.mean(simple[-50:]):.1f}")
print(f"Advanced Actor-Critic avg reward (last 50 eps): {np.mean(advanced[-50:]):.1f}")
print("Advantage subtracts the baseline V(s) -> lower variance gradients -> more stable learning.")
```

**Representative output (approximate)**
  Simple Actor-Critic   avg reward (last 50 eps): 115.5
  Advanced Actor-Critic avg reward (last 50 eps): 236.0

## Lab 12 — Proximal Policy Optimization

**Objective**
  Apply PPO (Proximal Policy Optimisation) to a real-world style task using a simulated
  environment: CartPole-v1 as a balance-control problem (Segway / inverted pendulum).
  Uses actor + critic, probability ratio and clipped surrogate objective.

**Libraries:** `gymnasium`, `torch`, `numpy`, `matplotlib`
**Approx. run time:** ~45 sec

```python
# --- 1. Setup: CartPole as a balance-control task (like a Segway / rocket landing balance) ---
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0); torch.manual_seed(0)

env = gym.make("CartPole-v1")
actor = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(dim=-1))
critic = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-3)
gamma, eps_clip = 0.99, 0.2
scores = []

for ep in range(1, 251):
    # --- 2. Collect one episode with the current (old) policy ---
    s, _ = env.reset(seed=ep)
    states, actions, old_log_probs, rewards = [], [], [], []
    done = False
    while not done:
        dist = torch.distributions.Categorical(actor(torch.FloatTensor(s)))
        a = dist.sample()
        next_s, r, term, trunc, _ = env.step(a.item())
        done = term or trunc
        states.append(s); actions.append(a); rewards.append(r)
        old_log_probs.append(dist.log_prob(a).detach())
        s = next_s
    scores.append(sum(rewards))

    # --- 3. Discounted returns ---
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    states_t = torch.FloatTensor(np.array(states))
    actions_t = torch.stack(actions)
    old_lp_t = torch.stack(old_log_probs)
    returns_t = torch.FloatTensor(returns)

    # --- 4. PPO update: 4 epochs of clipped surrogate loss, ratio = pi_new / pi_old ---
    for _ in range(4):
        values = critic(states_t).squeeze(-1)
        advantage = (returns_t - values).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        new_lp_t = torch.distributions.Categorical(actor(states_t)).log_prob(actions_t)
        ratio = torch.exp(new_lp_t - old_lp_t)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(values, returns_t)
        loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if ep % 25 == 0:
        print(f"PPO Episode {ep:3d} | Avg Reward (last 25): {np.mean(scores[-25:]):.1f}")

env.close()
plt.plot(scores); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("PPO on CartPole-v1"); plt.show()
```

**Representative output (approximate)**
  PPO Episode  25 | Avg Reward (last 25): 23.8
  PPO Episode 100 | Avg Reward (last 25): 130.1
  PPO Episode 175 | Avg Reward (last 25): 410.3
  PPO Episode 250 | Avg Reward (last 25): 500.0      (500 = perfect balance)

+ reward plot

---

*End of workbook*
