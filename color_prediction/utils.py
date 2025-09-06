import torch
import numpy as np
from skimage.color import lab2rgb
from torchvision.utils import save_image

def lab_to_rgb(L_channel, ab_channels):
    """
    Combines L and AB channels to produce an RGB image.
    Args:
      L_channel (torch.Tensor): Tensor of L-channel values. Shape (N, 1, H, W).
      ab_channels (torch.Tensor): Tensor of a and b channel values. Shape (N, 2, H, W).
    Returns:
      rgb_image (torch.Tensor): Tensor of RGB image. Shape (N, 3, H, W).
    """
    # Un-normalize L, a, and b channels
    L_channel = L_channel * 50.0 + 50.0
    ab_channels = ab_channels * 128.0
    
    # Combine L and AB channels
    lab_image = torch.cat([L_channel, ab_channels], dim=1).cpu().numpy()
    
    # Convert LAB to RGB
    rgb_imgs = []
    for img in lab_image:
        img_rgb = lab2rgb(np.transpose(img, (1, 2, 0)))
        rgb_imgs.append(img_rgb)
    
    return torch.from_numpy(np.stack(rgb_imgs, axis=0)).permute(0, 3, 1, 2).float()

def save_model(model, filename="colorization_model.pth"):
    """
    Saves the model state.
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model, filename="colorization_model.pth"):
    """
    Loads the model state.
    """
    try:
        model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")
    except FileNotFoundError:
        print(f"Model file {filename} not found. Starting from scratch.")