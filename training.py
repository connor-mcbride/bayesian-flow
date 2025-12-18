import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from unet_model import SimpleUNet
from datasets import get_cifar10_loaders
from utils import *


def discretized_loss(logits, target):
    """Discretized cross entropy
    
    logits: [B, 48, H, w]
    target: [B, 3, H, W], integer values 0..15
    """
    B, _, H, W = logits.shape
    levels = 16
    logits = logits.view(B, 3, levels, H, W).permute(0, 1, 3, 4, 2)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, levels),
        target.reshape(-1),
    )
    return loss


def bfn_categorical_loss(model, x4, device):
    B = x4.shape[0]
    t = torch.rand(B, device=device)

    q_t = make_qt_4bit(x4, t)
    q_t = add_time_channel(q_t, t)

    logits = model(q_t)
    logits = logits.view(B, 3, 16, *x4.shape[2:]).permute(0, 1, 3, 4, 2)

    ce = F.cross_entropy(
        logits.reshape(-1, 16),
        x4.reshape(-1),
        reduction="none"
    ).view(B, -1).mean(dim=1)

    loss = (ce / beta(t)).mean()

    return loss


def bfn_loss_modelB(modelB, x8, device):
    """
    modelB predicts x_lo (lower 4 bits) conditioned on x_hi (upper 4 bits)
    """
    B = x8.shape[0]
    x8 = x8.to(device)

    x_hi = (x8 // 16).long()
    x_lo = (x8 % 16).long()

    t = torch.rand(B, device=device)

    q_lo_t = make_qt_4bit(x_lo, t)

    q_hi = one_hot_4bit(x_hi)
    mu_hi, H_hi = hi_stats_from_qhi(q_hi)

    x_in = torch.cat([q_lo_t, q_hi, mu_hi, H_hi], dim=1)
    x_in = add_time_channel(x_in, t)

    logits = modelB(x_in)

    K = 16
    logits = logits.view(B, 3, K, *x_lo.shape[2:]).permute(0, 1, 3, 4, 2)

    ce = F.cross_entropy(
        logits.reshape(-1, K),
        x_lo.reshape(-1),
        reduction="none"
    ).view(B, -1).mean(dim=1)

    loss = (ce / beta(t)).mean()
    return loss


def train_modelA(train_loader, device, model, opt, num_epochs=50, plot=False):
    model.train()
    for epoch in range(num_epochs):
        for x8, _ in train_loader:
            x8 = x8.to(device)
            x4 = to_4bit(x8)

            loss = bfn_categorical_loss(model, x4, device)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Plot entropy near start, mid training, and the end
            if plot and epoch in [10, num_epochs // 2, num_epochs - 1]:
                times, H_in = measure_input_entropy(x4, device)
                _, H_out = measure_model_entropy(model, x4, device)
                plt.figure(figsize=(6,4))
                plt.plot(times, H_in, label="Input Entropy q_t")
                plt.plot(times, H_out, label="Model Output Entropy")
                plt.xlabel("Time t")
                plt.ylabel("Entropy")
                plt.legend()
                plt.title(f"Entropy vs Time (BFN) epoch={epoch}")

                plt.savefig(f"plots/model_input_entropy_{epoch}.png")

        print(f"Model A epoch {epoch}: train_loss={loss.item():.4f}")


def train_modelB(train_loader, device, model, opt, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        for x8, _ in train_loader:
            loss = bfn_loss_modelB(model, x8, device)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Model B epoch {epoch}: lossB={loss.item():.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, _ = get_cifar10_loaders()

    modelA = SimpleUNet(in_ch=49).to(device)
    modelB = SimpleUNet(in_ch=103).to(device)
    optA = torch.optim.AdamW(modelA.parameters(), lr=2e-4)
    optB = torch.optim.AdamW(modelB.parameters(), lr=2e-4)

    train_modelA(train_loader, device, modelA, optA, num_epochs=100)
    train_modelB(train_loader, device, modelB, optB, num_epochs=100)

    q_hi, x_hi = sample_modelA(modelA, device, B=16, steps=100)

    x_lo = sample_modelB(modelB, q_hi, device, B=16, steps=100)

    # Combine bits
    x8 = (16 * x_hi + x_lo).float() / 255.0
    save_image(x8, "samples/hierarchical_bfn.png", nrow=4)
