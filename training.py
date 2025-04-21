import random
import gymnasium as gym
import torch

from dqn import DQNCNN
from frame_stack import FrameStack
from memory_buffer import ReplayBuffer
from pre_processing import preprocess

from gymnasium.wrappers import RecordVideo

import os
from datetime import datetime

#################################################################
# The following is to suppress warnings...

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_folder = os.path.join("videos", f"run_{timestamp}")
os.makedirs(video_folder, exist_ok=True)

import shutil

if os.path.exists(video_folder):
    shutil.rmtree(video_folder)
os.makedirs(video_folder)

import warnings
warnings.filterwarnings("ignore", message=".*Overwriting existing videos.*")

#################################################################

render_mode="rgb_array"
env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
num_actions = 5

# Only record every Nth episode (e.g., every 50 episodes)
env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep: ep % 50 == 0)

obs, _ = env.reset()
gray = preprocess(obs)

# Buffer constants
buffer_capacity = 100_000
min_buffer_size = 5000
batch_size = 32

# Stack 4 frames at once
stacker = FrameStack(k=4)
memory_buffer = ReplayBuffer(capacity=buffer_capacity)
network_sync_rate = 5000 # How often to copy the policy network to the target network
num_episodes = 1000
num_steps = 0
eval_interval = 50 # How often to evaluate the policy network
eval_epsilon = 0.05 # Epsilon for evaluation

# Exploration values
epsilon = 1.0 # actually this will change
epsilon_end = 0.1
epsilon_decay = 0.995  # decay per step

# Future reward discount factor
gamma = 0.99

# Initialize the DQN network
policy_network = DQNCNN(action_size=env.action_space.n)
target_network = DQNCNN(action_size=env.action_space.n)
target_network.load_state_dict(policy_network.state_dict())
optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-4)

# Record reward
record_reward = float('-inf')
episode_rewards = []
evaluated_rewards = []

for episode in range(num_episodes):
    obs, _ = env.reset()                     # raw frame from env
    gray_scaled_image = preprocess(obs)      # shape: [1, 96, 96]
    state = stacker.reset(gray_scaled_image) # shape: [4, 96, 96] - copies the initial frame four times

    done = False
    episode_reward = 0
    while not done:
        # Choose action based on state (your stacked 4xHxW input)
        if random.random() < epsilon:
            action = random.randrange(num_actions) # Random action
        else:
            # Add a batch dimension to the state when we pass it in
            action = policy_network(state.unsqueeze(0)).argmax().item()  # Choose action with highest Q-value
        # Increment the steps taken
        num_steps += 1

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        gray = preprocess(obs)
        next_state = stacker.step(gray)

        # Store (state, action, reward, next_state, done) in replay buffer
        memory_buffer.push(state, action, reward, next_state, done)
        # NOTE that when we pass in the state, we don't need to add a batch dimension, because the replay buffer - when sampled - will spit out a batch of states (and other values)

        # Train your network if the buffer has enough samples
        if len(memory_buffer) >= min_buffer_size:
            # Sample a batch from the replay buffer
            states, actions, rewards, next_states, dones = memory_buffer.sample(batch_size)
            # Compute the Q-Values and Loss
            policy_q_values = policy_network(states) # shape is [batch_size, num_actions]
            # Which q-value do we pick out? The one corresponding to the action we took
            q_values = policy_q_values.gather(1, actions.unsqueeze(1)) # Pick out the one Q-Value (for each batch member) corresponding to the action actually taken in each sample - [batch_size, 1]
            q_values = q_values.squeeze(1) # [batch_size]
            with torch.no_grad():
                target_q_values = target_network(next_states)
                # Compute the max Q-value for the next state to use in the target Q-value for the current state
                max_next_q = target_q_values.max(1)[0]
                target_q_values = rewards + (1 - dones) * gamma * max_next_q 
            # Compute the loss
            loss = torch.nn.functional.mse_loss(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the state
        state = next_state
        # Check if the episode is done
        done = terminated or truncated

        if done:
            print(f"Episode reward: {episode_reward}")

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if not num_steps % network_sync_rate:
            # Copy the policy network to the target network
            target_network.load_state_dict(policy_network.state_dict())
    
    # Store the episode reward and potentially save the model
    episode_rewards.append(episode_reward)
    if episode_reward > record_reward:
        record_reward = episode_reward
        torch.save(policy_network.state_dict(), "best_model.pth")

    if not episode % eval_interval:
        render_mode="rgb_array"
        eval_env = RecordVideo(gym.make("CarRacing-v3", continuous=False, render_mode=render_mode), video_folder=video_folder, name_prefix=f"eval_ep{episode}", episode_trigger=lambda _: True)
        # Evaluate the policy network for clean performance stats - no gradient attached
        total_reward = 0
        done = False
        obs, _ = eval_env.reset()     # raw frame from env
        gray_scaled_image = preprocess(obs)      # shape: [1, 96, 96]
        state = stacker.reset(gray_scaled_image) # shape: [4, 96, 96] - copies the initial frame four times
        while not done:
            with torch.no_grad(): # Same as in the above training...
                # Now select the action based on epsilon greedy policy
                if random.random() < eval_epsilon:
                    action = random.randrange(num_actions) # Random action
                else:
                    # Add batch dimension to the state when we pass it in
                    action = policy_network(state.unsqueeze(0)).argmax().item()  # Choose action with highest Q-value (according to policy)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                gray = preprocess(obs)
                next_state = stacker.step(gray)
                # Update the state
                state = next_state
                # Check if the episode is done
                done = terminated or truncated
        evaluated_rewards.append(total_reward)

# Now we can plot the rewards
import matplotlib.pyplot as plt
plt.plot(episode_rewards, label='Episode Rewards')
plt.plot(evaluated_rewards, label='Evaluated Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode and Evaluated Rewards')
plt.legend()
plt.savefig('rewards.png')