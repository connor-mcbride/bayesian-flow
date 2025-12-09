import torch
from torchvision import datasets, transforms

def get_cifar10_loaders(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 255).to(torch.uint8))
    ])

    train = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    test = datasets.CIFAR10("data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader