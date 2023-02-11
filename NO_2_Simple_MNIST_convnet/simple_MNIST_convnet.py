import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 15
IMG_SIZE = (28, 28)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# datasets for training
train_ds = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# datasets for validation
val_ds = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
'''
此处没有考虑将labels设置成ont hot的形式，是因为用loss函数用CrossEntropyLoss()的target可以有两种形式
分别是：
target with class indices   
target with class probabilities
'''

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE)

for X, y in val_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# build the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(3136, 10),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.layers(x)
        return out


model = NeuralNetwork().to(device)
print(f'model:\n{model}')
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def mytrain(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)*BATCH_SIZE
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.type(torch.float32).to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")


def myval(dataloader, model, loss_fn):
    size = len(dataloader)*BATCH_SIZE
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(1)
            val_loss += loss_fn(pred, y).item()

            correct += torch.eq(pred.argmax(1), y).sum().item()
    val_loss /= num_batches
    correct /= size
    print(f"Val Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return round(correct, 2)



epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------")
    mytrain(train_dataloader, model, loss_fn, optimizer)
    myval(val_dataloader, model, loss_fn)
print("Done!")


