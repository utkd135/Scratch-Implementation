import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from ViT_module import ViT
from torch.utils.data import DataLoader
from tqdm import tqdm


# Loading MNIST Dataset
batchsize = 400
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
train_set = torchvision.datasets.MNIST(root='./', download=True, train=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./', download=True, train=False, transform=transform)

train_loader = DataLoader(train_set, batchsize, pin_memory=True, shuffle=True) # , num_workers=4)
test_loader = DataLoader(test_set, batchsize, pin_memory=True) # , num_workers=4)

model = ViT(1, 10, batchsize, device='cuda').to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
print(len(train_set), len(test_set))
print(len(train_loader), len(test_loader))

for epoch in range(epochs):
    train_loss = 0
    test_loss = 0
    correct = 0
    total = 0
    for img, label in tqdm(train_loader):
        img, label = img.to('cuda'), label.to('cuda')
        output = model(img)
        loss = criterion(output, label)
        train_loss += loss.detach().cpu().item() / len(train_loader)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.2f}")

    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to('cuda'), label.to('cuda')
            output = model(img)
            loss = criterion(output, label)
            test_loss += loss.detach().cpu().item() / len(test_loader)
            correct += torch.sum(torch.argmax(output, dim=-1) == label).detach().cpu()
            total += len(img)

    accuracy = (correct.numpy() / total) * 100
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {accuracy:.2f}%")