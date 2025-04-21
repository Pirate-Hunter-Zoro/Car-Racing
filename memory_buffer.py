import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),                  # [batch_size, C, H, W]
            torch.tensor(actions),                # [batch_size]
            torch.tensor(rewards, dtype=torch.float32),      # [batch_size]
            torch.stack(next_states),             # [batch_size, C, H, W]
            torch.tensor(dones, dtype=torch.float32)         # [batch_size]
        )

    def __len__(self):
        return len(self.buffer)
