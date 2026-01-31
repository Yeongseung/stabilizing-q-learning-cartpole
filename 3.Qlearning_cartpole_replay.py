import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# ==================================================
# Plot Rewards
# ==================================================
def plot_rewards(rewards, td_errors):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    rewards = np.array(rewards)
    td_errors = np.array(td_errors)

    # 1. Rewards Plot
    ax1.plot(rewards, label='Total Reward', alpha=0.4)
    if len(rewards) >= 100:
        means = [np.mean(rewards[max(0, i-100):(i+1)]) for i in range(len(rewards))]
        ax1.plot(means, label='Moving Average (100 eps)', color='red')
    
    if max(rewards) >= 500:
        first_max_indices = np.where(rewards >= 500)[0]
        if len(first_max_indices) > 0:
            first_max_idx = first_max_indices[0]
            ax1.axvline(x=first_max_idx, color='green', linestyle='--', label=f'First Max (Ep {first_max_idx})')

    ax1.set_title('Training Progress: Total Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)

    # 2. TD Error Plot
    # set the basic color and find the index of the point where the reward is 500.
    colors = ['orange' if r < 500 else 'blue' for r in rewards]
    
    # to represent the data like a bar graph, we can consider using stem or bar, but since the data is many, it is the most clean to combine scatter and line.
    ax2.plot(td_errors, color='gray', alpha=0.2) # background of the whole flow
    
    # Ensure indices don't exceed td_errors length
    max_idx = len(td_errors) - 1
    
    # data with reward < 500 (orange)
    low_indices = np.where(rewards < 500)[0]
    low_indices = low_indices[low_indices <= max_idx]  # Filter valid indices
    if len(low_indices) > 0:
        ax2.scatter(low_indices, td_errors[low_indices], color='orange', s=2, alpha=0.5, label='Reward < 500')
    
    # data with reward >= 500 (blue)
    high_indices = np.where(rewards >= 500)[0]
    high_indices = high_indices[high_indices <= max_idx]  # Filter valid indices
    if len(high_indices) > 0:
        ax2.scatter(high_indices, td_errors[high_indices], color='blue', s=10, alpha=0.8, label='Reward = 500')

    ax2.set_title('TD Error Trend (Blue dots: Max Reward episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Square Error')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# ==================================================
# Plot Q-Values for Fixed States
# ==================================================
def plot_q_values(q_value_history, state_labels):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(5):
        q_history = np.array(q_value_history[i])
        
        # Q-value for each action (0: left, 1: right)
        q_left = q_history[:, 0]
        q_right = q_history[:, 1]
        
        axes[i].plot(q_left, label='Q(s, Left)', alpha=0.7, color='blue')
        axes[i].plot(q_right, label='Q(s, Right)', alpha=0.7, color='red')
        axes[i].set_title(f'{state_labels[i]}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel('Q-value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide the 6th subplot (we only have 5 states)
    axes[5].axis('off')
    
    plt.suptitle('Q-value Evolution for Fixed States', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==================================================
# Experience Replay Buffer
# ==================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        """Add experience (state, action, reward, next_state, done) to buffer"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample batch_size experiences from buffer"""
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.buffer)

# ==================================================
# Q-Network
# ==================================================
class qnetwork(torch.nn.Module):
    def __init__(self, state_size:4, action_size:2):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# ==================================================


# Main
# ==================================================
model = qnetwork(4,2)
env = gym.make("CartPole-v1", render_mode="none")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
all_rewards = [] # to plot rewards
all_td_errors = []

# Experience Replay Buffer
replay_buffer = ReplayBuffer(capacity=10000)
batch_size = 32
update_frequency = 4  # Train every 4 steps

# fixed states to observe
fixed_states = [
    np.array([2.0, 0.0, 0.0, 0.0]),    # 1.     right edge (x=2.3 / Limit 2.4)
    np.array([0.0, 0.0, 0.15, 0.0]),     # 2. leaning right (theta=0.2 / Limit 0.2095)
    np.array([0.0, 0.0, 0.0, 0.0]),    # 3. stable (center, vertical)
    np.array([-2.0, 0.0, 0.0, 0.0]),  # 4. left edge (x=-2.3 / Limit -2.4)
    np.array([0.0, 0.0, -0.15, 0.0])   # 5. leaning left (theta=-0.2 / Limit -0.2095)
]
state_labels = ["Edge Right", "Leaning Right", "Stable", "Edge Left", "Leaning Left"]
q_value_history = [[] for _ in range(5)]

for episode in range(3000): # 5000 episodes
    # ==================================================
    # Reset
    # ==================================================
    obs, info = env.reset()
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    game_over = False
    total_reward = 0
    episode_losses = []
    step_count = 0

    action_space = [0,1] # 0: left, 1: right

    while not game_over:
        state_tensor = torch.tensor(obs, dtype=torch.float32)
        q_values = model(state_tensor)

        if np.random.random() < 0.1: # epsilon = 0.1
            act = np.random.choice(action_space)
        else:
            act = torch.argmax(q_values).item()

        # step the environment
        next_obs, reward, terminated, truncated, info = env.step(act)
        total_reward += reward

        # Experience Replay
        replay_buffer.add(obs.copy(), act, reward, next_obs.copy(), terminated)

        # Experience Replay (Batch Training)
        if len(replay_buffer) >= batch_size and step_count % update_frequency == 0:
            batch = replay_buffer.sample(batch_size)
            if batch is not None:
                # prepare the batch data
                states = torch.tensor([e[0] for e in batch], dtype=torch.float32)
                actions = torch.tensor([e[1] for e in batch], dtype=torch.long)
                rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32)
                next_states = torch.tensor([e[3] for e in batch], dtype=torch.float32)
                terminateds = torch.tensor([e[4] for e in batch], dtype=torch.float32)

                # current q_values
                current_q_values = model(states)
                current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # target q_values (Q-Learning: max Q(s', a'))
                # terminated only check - truncated is okay because we can still get the next reward.
                with torch.no_grad():
                    next_q_values = model(next_states)
                    next_q_max = torch.max(next_q_values, dim=1)[0]
                    target_q = rewards + (0.99 * next_q_max * (1 - terminateds))

                # calculate the loss and update the model
                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_losses.append(loss.item())

        obs = next_obs # update the state to the next state
        step_count += 1

        if terminated or truncated:
            game_over = True
            all_rewards.append(total_reward)

            # Ensure all_rewards and all_td_errors have the same length
            if episode_losses:
                all_td_errors.append(np.mean(episode_losses))
            else:
                all_td_errors.append(0.0)  # No loss if episode was too short

            # track the q_values for the fixed states
            for i, fixed_state in enumerate(fixed_states):
                with torch.no_grad():
                    fixed_state_tensor = torch.tensor(fixed_state, dtype=torch.float32)
                    q_vals = model(fixed_state_tensor).numpy()
                    q_value_history[i].append(q_vals)

            print(f"episode: {episode}, Total reward: {total_reward}")
            break
        

env.close()
plot_rewards(all_rewards, all_td_errors)
plot_q_values(q_value_history, state_labels)
