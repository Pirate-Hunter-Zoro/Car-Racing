from collections import deque
import torch

class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, initial_frame):
        # Just create four copies of the initial frame
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(initial_frame)
        return self._get_stack()

    def step(self, new_frame):
        self.frames.append(new_frame)
        return self._get_stack()

    def _get_stack(self):
        return torch.cat(list(self.frames), dim=0)  # Shape: [k, H, W] - k frames, each with height and width