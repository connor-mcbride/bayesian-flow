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


def one_hot_4bit(x4, K=16):
    """One-hot encode each pixel/channel"""
    # x4: [B,3,H,W] integer 0..15
    B, C, H, W = x4.shape
    oh = F.one_hot(x4, num_classes=K).permute(0, 1, 4, 2, 3).float()
    return oh.reshape(B, C*K, H, W)


def probs_4bit_from_48(q48, K=16):
    B, _, H, W = q48.shape
    return q48.view(B, 3, K, H, W).permute(0, 1, 3, 4, 2)


def hi_stats_from_qhi(q_hi_48, K=16):
    p = probs_4bit_from_48(q_hi_48, K=K)
    k = torch.arange(K, device=q_hi_48.device).view(1, 1, 1, 1, K)
    mu = (p * k).sum(dim=-1)
    ent = categorical_entropy(p)
    mu = mu / (K - 1)

    return mu, ent


@torch.no_grad()
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


@torch.no_grad()
def sample_bfn(model, device, B=16, steps=100):
    model.eval()
    C, H, W = 3, 32, 32
    K = 16

    q = torch.full((B, C, K, H, W), 1.0 / K, device=device)

    ts = torch.linspace(0, 1, steps, device=device)

    for i in range(steps - 1):
        t = ts[i].expand(B)

        q_in = q.permute(0, 1, 3, 4, 2).reshape(B, C*K, H, W)
        q_in = add_time_channel(q_in, t)

        logits = model(q_in)
        logits = logits.view(B, C, K, H, W)

        p = torch.softmax(logits, dim=2)

        dt = ts[i+1] - ts[i]
        q = q + dt * (p-q)

        q = q.clamp(min=1e-6)
        q = q / q.sum(dim=2, keepdim=True)

    q_final = q.permute(0, 1, 3, 4, 2)
    x4 = torch.distributions.Categorical(q_final).sample()

    x8 = (x4 * 16).float() / 255.0
    return x8


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
    return 0.1 + 0.9 * t


def make_qt_4bit(x4, t, K=16):
    """    
    :param x4: [B,3,H,W] integer ground truth
    :param t: [B]
    returns: q_t [B, 48, H, W]
    """
    x_onehot = one_hot_4bit(x4, K=K)
    uniform = torch.full_like(x_onehot, 1.0 / K)
    a = alpha(t).view(-1, 1, 1, 1)

    return (1-a) * uniform + a * x_onehot


def categorical_entropy(p, eps=1e-8):
    """    
    :param p: [..., K] categorical probabilities
    :param eps: entropy [...]
    """
    return -(p * (p + eps).log()).sum(dim=-1)


@torch.no_grad()
def measure_input_entropy(x4, device, num_t=50):
    """
    Docstring for measure_input_entropy
    
    :param x4: [B,3,H,W] ground truth
    returns: times, entropies
    """
    B = x4.shape[0]
    K = 16
    times = torch.linspace(0, 1, num_t, device=device)

    entropies = []

    for t in times:
        t_batch = torch.full((B,), t, device=device)

        q_t = make_qt_4bit(x4, t_batch, K=K)
        q_t = q_t.view(B, 3, K, *x4.shape[2:]) \
                 .permute(0,1,3,4,2)
        
        H = categorical_entropy(q_t)
        entropies.append(H.mean().item())

    return times.cpu(), entropies


@torch.no_grad()
def measure_model_entropy(model, x4, device, num_t=50):
    model.eval()
    B = x4.shape[0]
    K = 16
    times = torch.linspace(0, 1, num_t, device=device)

    entropies = []

    for t in times:
        t_batch = torch.full((B,), t, device=device)

        q_t = make_qt_4bit(x4, t_batch, K=K)
        q_t = add_time_channel(q_t, t_batch)

        logits = model(q_t)
        logits = logits.view(B, 3, K, *x4.shape[2:]) \
                       .permute(0, 1, 3, 4, 2)
        
        probs = torch.softmax(logits, dim=-1)

        H = categorical_entropy(probs)
        entropies.append(H.mean().item())

    return times.cpu(), entropies
