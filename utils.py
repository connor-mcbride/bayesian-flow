import torch
import torch.nn.functional as F

def to_4bit(x8):
    """
    Converts images to 4-bit
    (Targets are integers 0-15 per pixel)
    
    :param x8: uint8 image: [B,3,H,W], values 0..255
    return: int image: values 0..15
    """
    return (x8 // 16).long()

def normalize_4bit(x4):
    """Normalize 4-bit values ot [0,1]"""
    return x4.float() / 15.0

def one_hot_4bit(x4):
    """One-hot encode each pixel/channel"""
    # x4: [B,3,H,W] integer 0..15
    return F.one_hot(x4, num_classes=16) \
        .permute(0,1,4,2,3) \
        .reshape(x4.shape[0], 3*16, x4.shape[2], x4.shape[3]) \
        .float()
