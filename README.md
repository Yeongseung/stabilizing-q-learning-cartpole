# Stabilizing Q-Learning in CartPole

This repository contains the implementation code for the blog post: **[Stabilizing Q-Learning in CartPole](https://yeongseung.github.io/posts/mastering_cartpole/post/)**

## Overview

This project explores various techniques to stabilize Q-Learning in the CartPole-v1 environment, addressing the periodic performance drops commonly observed in reinforcement learning. Through systematic experiments, we demonstrate how techniques like Experience Replay, Double DQN, and specialized buffer strategies can improve training stability.

## Experiments

The code is organized into 7 main files, each representing a progressive improvement in the algorithm:

### 1. SARSA (`1.SARSA_cartpole.py`)
- Baseline implementation using the SARSA algorithm
- On-policy learning with epsilon-greedy action selection
- Demonstrates the fundamental challenge: periodic drops in performance

### 2. Q-Learning (`2.Qlearning_cartpole.py`)
- Off-policy Q-Learning implementation
- Uses greedy action selection for target Q-value calculation
- Similar instability to SARSA

### 3. Q-Learning with Experience Replay (`3.Qlearning_cartpole_replay.py`)
- Introduces vanilla Experience Replay
- Buffer capacity: 10,000
- Batch size: 32, Update frequency: every 4 steps
- Improves stability but Q-values grow uncontrollably

### 4. Dual Buffer System (`4.Qlearning_cartpole_replay_dual.py`)
- Separates experiences into success and failed buffers
- Success buffer: episodes with reward ≥ 500
- Failed buffer: episodes with reward < 500
- Sampling ratio: 7:3 (success:failed)
- Prevents distribution narrowing but doesn't fully solve instability

### 5. Double DQN with Dual Buffer (`5.Qlearning_cartpole_replay_dual_double.py`)
- Adds target network to prevent Q-value overestimation
- Soft update: τ = 0.01 (1% per step)
- Significantly stabilizes Q-values and TD error
- **Key improvement**: Mitigates sudden Q-value spikes

### 5.1 Double DQN without Dual Buffer (`5_1.Qlearning_cartpole_replay_double.py`)
- Comparison variant: Double DQN with single buffer
- Buffer capacity: 10,000
- Used to demonstrate the effectiveness of the dual buffer strategy

### 6. Failure Buffer with Triple Buffer System (`6.Qlearning_cartpole_replay_dual_double_failure.py`)
- Introduces three buffers:
  - **Temporary buffer**: All experiences (capacity: 10,000)
  - **Success buffer**: Successful episodes (capacity: 5,000)
  - **Failure buffer**: Only terminal states where reward = 0 (capacity: 5,000)
- Sampling: 7:3 ratio initially (temporary:failed), then 7:0:3 (success:temporary:failed)
- Focuses learning on the most informative failure signals
- Tracks failure categories: Edge Left/Right, Leaning Left/Right

### 7. Normal + Failure Buffer (`7.Qlearning_cartpole_replay_normal_failure.py`)
- Simplified to two buffers:
  - **Temporary buffer**: All experiences (capacity: 10,000)
  - **Failure buffer**: Only terminal states (capacity: 5,000)
- Sampling ratio: 7:3 (temporary:failed)
- Enables incremental learning from failure to success
- **Best overall stability** in experiments

## Requirements

```
torch
gymnasium
numpy
matplotlib
```

## Usage

Run any experiment file directly:

```bash
python 1.SARSA_cartpole.py
python 2.Qlearning_cartpole.py
python 3.Qlearning_cartpole_replay.py
python 4.Qlearning_cartpole_replay_dual.py
python 5.Qlearning_cartpole_replay_dual_double.py
python 5_1.Qlearning_cartpole_replay_double.py
python 6.Qlearning_cartpole_replay_dual_double_failure.py
python 7.Qlearning_cartpole_replay_normal_failure.py
```

Each script will:
1. Train the agent for 3,000 episodes
2. Display training progress in the console
3. Show visualization plots after training:
   - Episode total reward and moving average
   - TD error trends
   - Q-value evolution for fixed states

## Key Findings

1. **Achieving reward 500 is relatively easy**, but maintaining stability is challenging
2. **Experience Replay alone causes Q-value explosion** without target network separation
3. **Double DQN is essential** when using Experience Replay to prevent overestimation
4. **Failure-focused sampling** significantly improves long-term stability by preserving rare but informative failure signals
5. **Buffer composition matters**: Balancing successful and failed experiences prevents both catastrophic forgetting and overfitting to success
6. **Failure diversity** is critical - the agent must encounter all failure modes (edge left/right, leaning left/right) to become truly robust

## Network Architecture

All experiments use the same Q-network architecture:

```python
class qnetwork(torch.nn.Module):
    def __init__(self, state_size=4, action_size=2):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

- Input: 4 state variables (cart position, cart velocity, pole angle, pole angular velocity)
- Hidden layer: 128 neurons with ReLU activation
- Output: 2 Q-values (left action, right action)

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Episodes | 3,000 | Total training episodes |
| Epsilon | 0.1 | Exploration rate (fixed) |
| Gamma | 0.99 | Discount factor |
| Learning Rate | 0.001 | Adam optimizer |
| Batch Size | 32 | Experience replay batch size |
| Update Frequency | 4 | Steps between model updates |
| Tau (Double DQN) | 0.01 | Soft update coefficient |

## Blog Post

For detailed explanations, visualizations, and insights, please read the full blog post:
**[Stabilizing Q-Learning in CartPole](https://yeongseung.github.io/posts/mastering_cartpole/post/)**

## License

This project is open source and available under the MIT License.

## Author

Yeongseung - [Blog](https://yeongseung.github.io/) | [GitHub](https://github.com/yeongseung)

## Citation

If you find this work helpful, please cite the blog post:

```
Yeongseung. (2026). Stabilizing Q-Learning in CartPole. 
Retrieved from https://yeongseung.github.io/posts/mastering_cartpole/post/
```

