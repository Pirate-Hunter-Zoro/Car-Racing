import torch
import torchvision.transforms.functional as TF

def preprocess(obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0  # [3, H, W] instead of [H,W,3]
    gray = TF.rgb_to_grayscale(obs_tensor)  # [1, H, W]
    return gray