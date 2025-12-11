import torch 

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