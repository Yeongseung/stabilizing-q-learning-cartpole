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
    colors = ['orange' if r < 500 else 'blue' for r in rewards]
    
    ax2.plot(td_errors, color='gray', alpha=0.2)
    
    low_indices = np.where(rewards < 500)[0]
    ax2.scatter(low_indices, td_errors[low_indices], color='orange', s=2, alpha=0.5, label='Reward < 500')
    
    high_indices = np.where(rewards >= 500)[0]
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

# fixed states to observe
fixed_states = [
    np.array([2.0, 0.0, 0.0, 0.0]),    # 1. Right edge (x=2.0 / Limit 2.4)
    np.array([0.0, 0.0, 0.15, 0.0]),   # 2. Leaning right (theta=0.15 / Limit 0.2095)
    np.array([0.0, 0.0, 0.0, 0.0]),    # 3. Perfect equilibrium (center, vertical)
    np.array([-2.0, 0.0, 0.0, 0.0]),   # 4. Left edge (x=-2.0 / Limit -2.4)
    np.array([0.0, 0.0, -0.15, 0.0])   # 5. Leaning left (theta=-0.15 / Limit -0.2095)
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

    action_space = [0,1] # 0: left, 1: right

    while not game_over:
        state_tensor = torch.tensor(obs, dtype=torch.float32)
        q_values = model(state_tensor)
        # no matter whether the action will be random or not, we get the q_values, up here.
        # But why? doesn't we use the q_values for the action selection only when exploiting?
        # Because, even if the action will be random, we update the model using the selected action's q_value.

        if np.random.random() < 0.1: # epsilon = 0.1
            act = np.random.choice(action_space)
        else:
            act = torch.argmax(q_values).item()
        
        # current_q is where the backpropagation starts.
        current_q = q_values[act]

        # step the environment
        next_obs, reward, terminated, truncated, info = env.step(act)
        total_reward += reward

        # calculate the target
        with torch.no_grad(): # to save memory and block the gradient
            next_state_tensor = torch.tensor(next_obs, dtype=torch.float32)
            next_q_max = torch.max(model(next_state_tensor))
            target_q = reward + (0.99 * next_q_max * (1 - terminated)) 
            # no next reward when terminated
            # But, truncated is okay. because we can still get the next reward.

        # calculate the loss and update the model
        loss = criterion(current_q, target_q)
        #target_q here is like the label for the current_q.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_losses.append(loss.item()) # Loss(TD Error)

        obs = next_obs # update the state to the next state

        if terminated or truncated:
            game_over = True
            all_rewards.append(total_reward)

            if episode_losses:
                all_td_errors.append(np.mean(episode_losses))

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
