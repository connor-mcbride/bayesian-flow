import torch
import torch.nn.functional as F
from torchvision.utils import save_image

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

@torch.no_grad
def sample(model, device, output="samples.png", B=16):
    model.eval()

    C, H, W = 3, 32, 32
    levels = 16

    x4_rand = torch.randint(
        0, levels, (B, C, H, W), device=device
    )
    q = one_hot_4bit(x4_rand)
    t = torch.ones(B, device=device)
    q_t = add_time_channel(q, t)
    logits = model(q_t)

    logits = logits.view(B, C, levels, H, W)
    probs = F.softmax(logits, dim=2)

    x4_hat = torch.distributions.Categorical(
        probs.permute(0, 1, 3, 4, 2)
    ).sample()
    x8_hat = (x4_hat * 16).clamp(0, 255).float() / 255.0

    save_image(x8_hat, output, nrow=int(B**0.5))

    print(f"Saved samples to {output}")

def add_time_channel(x, t):
    """
    x: [B, C, H, W]
    t: [B]
    returns: [B, C+1, H, W]
    """
    B, _, H, W = x.shape
    t_channel = t.view(B, 1, 1, 1).expand(B, 1, H, W)
    return torch.cat([x, t_channel], dim=1)

def alpha(t):
    return t

def beta(t):
    return t + 1e-4

def make_qt(x4, t, K=16):
    """
    Docstring for make_qt
    
    :param x4: [B,3,H,W] integer ground truth
    :param t: [B]
    returns: q_t [B, 48, H, W]
    """
    B, C, H, W = x4.shape

    x_onehot = F.one_hot(x4, K) \
        .permute(0,1,4,2,3) \
        .reshape(B, C*K, H, W) \
        .float()
    
    uniform = torch.full_like(x_onehot, 1.0 / K)

    a = alpha(t).view(B, 1, 1, 1)

    return (1-a) * uniform + a * x_onehot