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
# Double DQN: Main network and Target network
model = qnetwork(4,2)
target_model = qnetwork(4,2)
# Target network initialization: Copy main network weights
target_model.load_state_dict(model.state_dict())
target_model.eval()  # Keep target network in evaluation mode

env = gym.make("CartPole-v1", render_mode="none")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
all_rewards = [] # to plot rewards
all_td_errors = []

# Double DQN: Soft update coefficient
tau = 0.01  # Soft update coefficient (0.01 = 1% update per step)

# Triple Buffer System
# 1. Temporary buffer: Store all experiences (immediately at each step)
# 2. Success buffer: Store all experiences from successful episodes
# 3. Failed buffer: Store only the last state before failure (1 per failure)
temporary_buffer = ReplayBuffer(capacity=10000)  # Store all experiences
success_buffer = ReplayBuffer(capacity=5000)  # Episodes with reward >= 500
failed_buffer = ReplayBuffer(capacity=5000)   # Only the state before failure
batch_size = 32
update_frequency = 4  # Train every 4 steps
success_ratio = 0.7  # 7:3 ratio (success:failed)

# Track failure categories
failed_categories = {
    "Edge Left": 0,      # Cart position <= -2.3
    "Edge Right": 0,     # Cart position >= 2.3
    "Leaning Left": 0,   # Pole angle <= -0.1995
    "Leaning Right": 0   # Pole angle >= 0.1995
}

def classify_failure_reason(state):
    """Classify failure reason into categories
    state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    """
    cart_pos = state[0]
    pole_angle = state[2]
    
    # Check cart position (lenient: -2.0, 2.0)
    if cart_pos <= -2.0:
        return "Edge Left"
    elif cart_pos >= 2.0:
        return "Edge Right"
    
    # Check pole angle (lenient: -0.1495, 0.1495)
    if pole_angle <= -0.1495:
        return "Leaning Left"
    elif pole_angle >= 0.1495:
        return "Leaning Right"

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
    episode_experiences = []  # Temporary: Store episode experiences
    prev_obs = None  # Track the state before failure

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

        # Temporary buffer: Store all experiences immediately
        temporary_buffer.add(obs.copy(), act, reward, next_obs.copy(), terminated)
        episode_experiences.append((obs.copy(), act, reward, next_obs.copy(), terminated))
        
        # Track the state before failure (only terminated is failure, not truncated)
        if terminated:
            # Store the state before failure (current obs is the state before failure)
            prev_obs = obs.copy()

        # Triple Buffer Experience Replay (Batch Training)
        total_buffer_size = len(temporary_buffer) + len(success_buffer) + len(failed_buffer)
        if total_buffer_size >= batch_size and step_count % update_frequency == 0:
            batch = []
            
            if len(success_buffer) == 0:
                # Initial: sample with 7:3 ratio (temporary:failed)
                temp_samples = int(batch_size * success_ratio)
                failed_samples = batch_size - temp_samples
                
                # Sample from temporary buffer
                if len(temporary_buffer) >= temp_samples:
                    temp_batch = temporary_buffer.sample(temp_samples)
                    if temp_batch:
                        batch.extend(temp_batch)
                else:
                    temp_batch = temporary_buffer.sample(len(temporary_buffer))
                    if temp_batch:
                        batch.extend(temp_batch)
                
                # Sample from failed buffer
                remaining = batch_size - len(batch)
                if remaining > 0 and len(failed_buffer) > 0:
                    if len(failed_buffer) >= remaining:
                        failed_batch = failed_buffer.sample(remaining)
                        if failed_batch:
                            batch.extend(failed_batch)
                    else:
                        failed_batch = failed_buffer.sample(len(failed_buffer))
                        if failed_batch:
                            batch.extend(failed_batch)
            else:
                # Later: sample with 7:0:3 ratio (successful:temporary:failed)
                success_samples = int(batch_size * success_ratio)
                failed_samples = batch_size - success_samples
                
                # Sample from success buffer
                if len(success_buffer) >= success_samples:
                    success_batch = success_buffer.sample(success_samples)
                    if success_batch:
                        batch.extend(success_batch)
                else:
                    success_batch = success_buffer.sample(len(success_buffer))
                    if success_batch:
                        batch.extend(success_batch)
                
                # Sample from failed buffer
                remaining = batch_size - len(batch)
                if remaining > 0 and len(failed_buffer) > 0:
                    if len(failed_buffer) >= remaining:
                        failed_batch = failed_buffer.sample(remaining)
                        if failed_batch:
                            batch.extend(failed_batch)
                    else:
                        failed_batch = failed_buffer.sample(len(failed_buffer))
                        if failed_batch:
                            batch.extend(failed_batch)
            
            # Train when batch is sufficient
            if len(batch) >= batch_size:
                # Use exactly batch_size samples
                batch = batch[:batch_size]
                # prepare the batch data
                states = torch.tensor([e[0] for e in batch], dtype=torch.float32)
                actions = torch.tensor([e[1] for e in batch], dtype=torch.long)
                rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32)
                next_states = torch.tensor([e[3] for e in batch], dtype=torch.float32)
                terminateds = torch.tensor([e[4] for e in batch], dtype=torch.float32)

                # current q_values (use main network)
                current_q_values = model(states)
                current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # target q_values (Double DQN: use target network)
                with torch.no_grad():
                    # Double DQN: Calculate Q-value of next state using target network
                    next_q_values = target_model(next_states)
                    next_q_max = torch.max(next_q_values, dim=1)[0]
                    target_q = rewards + (0.99 * next_q_max * (1 - terminateds))

                # calculate the loss and update the model
                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_losses.append(loss.item())
                
                # Double DQN: Soft update (update a small portion at each learning step)
                with torch.no_grad():
                    for target_param, main_param in zip(target_model.parameters(), model.parameters()):
                        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

        obs = next_obs # update the state to the next state
        step_count += 1

        if terminated or truncated:
            game_over = True
            all_rewards.append(total_reward)
            
            # Initialize failure category
            failure_category = None

            # Triple Buffer: Store experiences in appropriate buffer based on total_reward
            if total_reward >= 500:
                # Success buffer: Store all experiences from successful episode (regardless of truncated or terminated)
                for exp in episode_experiences:
                    success_buffer.add(*exp)
            elif terminated:  # total_reward < 500 and terminated: this is a failure
                # Failed buffer: Store only the last state before failure (1 per failure)
                if len(episode_experiences) > 0:
                    # Store the last experience (state before failure) in failed buffer
                    last_exp = episode_experiences[-1]
                    failed_state = last_exp[0]  # State before failure
                    
                    # Classify and record failure reason
                    failure_category = classify_failure_reason(failed_state)
                    failed_categories[failure_category] += 1
                    
                    failed_buffer.add(*last_exp)
            # If truncated and total_reward < 500: Do not store (max steps reached but not a failure)

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

            # Print failure category if failed
            if failure_category is not None:
                print(f"episode: {episode}, Total reward: {total_reward}, Temp: {len(temporary_buffer)}, Success: {len(success_buffer)}, Failed: {len(failed_buffer)}, Reason: {failure_category}")
            else:
                print(f"episode: {episode}, Total reward: {total_reward}, Temp: {len(temporary_buffer)}, Success: {len(success_buffer)}, Failed: {len(failed_buffer)}")
            break
        

env.close()

# Print failed buffer statistics by category
print("\n" + "="*60)
print("Failed Buffer Statistics by Category")
print("="*60)
print(f"Total failed experiences: {len(failed_buffer)}")
for category, count in failed_categories.items():
    percentage = (count / len(failed_buffer) * 100) if len(failed_buffer) > 0 else 0
    print(f"  {category}: {count} ({percentage:.1f}%)")
print("="*60 + "\n")

plot_rewards(all_rewards, all_td_errors)
plot_q_values(q_value_history, state_labels)
