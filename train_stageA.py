import torch
from torchvision.utils import save_image
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


def train(train_loader, test_loader, device, model, opt, num_epochs=50):
    for epoch in range(num_epochs):
        for x8, _ in train_loader:
            x8 = x8.to(device)
            x4 = to_4bit(x8)
            q = one_hot_4bit(x4)

            B = x4.shape[0]
            t = torch.rand(B, device=device)
            q_t = add_time_channel(q, t)

            logits = model(q_t)

            loss = discretized_loss(logits, x4)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"epoch {epoch}: train_loss={loss.item():.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = get_cifar10_loaders()

    model = SimpleUNet(in_ch=49).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    train(train_loader, test_loader, device, model, opt, num_epochs=50)

    sample(model, device, "samples/4_bit_discrete_samples.png")
