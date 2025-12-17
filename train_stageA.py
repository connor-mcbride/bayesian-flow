import torch
import matplotlib.pyplot as plt
from model_stageA import SimpleUNet
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

    q_t = make_qt(x4, t)
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


def train(train_loader, test_loader, device, model, opt, num_epochs=50, plot=False):
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
                x8 = next(train_loader)
                x4 = to_4bit(x8)
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

        print(f"epoch {epoch}: train_loss={loss.item():.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = get_cifar10_loaders()

    model = SimpleUNet(in_ch=49).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    train(train_loader, test_loader, device, model, opt, num_epochs=50, plot=True)

    sample(model, device, "samples/4_bit_bfn_loss.png")
