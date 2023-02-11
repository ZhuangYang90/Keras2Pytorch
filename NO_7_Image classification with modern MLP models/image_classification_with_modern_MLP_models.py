import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from my_utils.process import image_extract_pathches

# Configure the hyperparameters
weight_decay = 0.0001
batch_size = 256
num_epochs = 1  # 原本为 50
dropout_rate = 0.2
image_size = 64  # We'll resize input images to this size.
patch_size = 8  # Size of the patches to be extracted from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
embedding_dim = 256  # Number of hidden units.
num_blocks = 4  # Number of blocks.
num_classes = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cup")

print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")
print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")

train_datasets = datasets.CIFAR100('E:/IMAGE/DataSet/CIFAR100',
                                   train=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize((image_size, image_size)),
                                       transforms.RandomResizedCrop(image_size),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                                   download=True)

val_datasets = datasets.CIFAR100('E:/IMAGE/DataSet/CIFAR100',
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Resize((image_size, image_size)),
                                     transforms.RandomResizedCrop(image_size),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                                 download=True)


train_dataloader = DataLoader(train_datasets, batch_size=batch_size)
val_dataloader = DataLoader(val_datasets, batch_size=batch_size)

class Patches(nn.Module):  # 目前只设计了h, w相等的方式，后面再研究不等的
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        h, w = x.size()[-2:]
        assert h == w
        kernel_size = h // self.patch_size
        x = image_extract_pathches(x, kernel_size=kernel_size, stride=kernel_size)
        return x


class BuildClassifier(nn.Module):
    def __init__(self, blocks, positional_encoding=False):
        super(BuildClassifier, self).__init__()
        self.positional_encoding = positional_encoding
        self.patches_fn = Patches(patch_size)
        self.linear = nn.Linear(192, embedding_dim)  # embedding=256
        self.blocks = blocks
        self.position_embedding_fn = nn.Embedding(
            num_embeddings=num_patches, embedding_dim=embedding_dim
        )

        self.category = nn.Linear(embedding_dim , num_classes)

    def forward(self, x):  # (B, 3, 64, 64)
        x = self.patches_fn(x)  # (B, 3*64, 8*8) -> (B, 192, 64)
        x = x.view(-1, 64, 192)
        x = self.linear(x)  # (B, 64, 256)
        if self.positional_encoding:
            positions = torch.arange(0, num_patches, step=1).type(torch.int64).to(device)  # num_patches=64
            position_embedding = self.position_embedding_fn(positions)  # (None, 64, 256)
            x = x + position_embedding  # (B, 64, 256)
        # Process x using the module blocks.
        x = self.blocks(x)  # (B, 64, 256)
        x = x.mean(dim=1)  # (B, 256)  相当于layers.GlobalAveragePooling1D()
        x = self.category(x)  # (B, 100)
        return x


class MLPMixerlayer(nn.Module):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super(MLPMixerlayer, self).__init__()
        self.normalize = nn.LazyBatchNorm1d(eps=1e-6)
        self.mlp1 = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Linear(num_patches, num_patches),
            nn.Dropout(p=dropout_rate)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(embedding_dim, num_patches),
            nn.GELU(),
            nn.Linear(num_patches, hidden_units),  # hidden_units = embedding_dim
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs):  # (B, 64, 256)
        x = self.normalize(inputs)  # (B, 64, 256)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = x.permute(0, 2, 1).contiguous()  # (B, 256, 64)
        mlp1_outputs = self.mlp1(x_channels)  # (B, 256, 64)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units]
        mlp1_outputs = mlp1_outputs.permute(0, 2, 1).contiguous()  # (B, 64, 256)
        # Add skip connection.
        x = inputs + mlp1_outputs  # (B, 64, 256)
        # Apply layer normalization.
        x_patches = self.normalize(x)  # (B, 64, 256)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)  # (B, 64, 256)
        # Add skip connection.
        x = x + mlp2_outputs  # (B, 64, 256)
        return x  # (B, 64, 256)


class FNetLayer(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super(FNetLayer, self).__init__()
        self.normlize1 = nn.LazyBatchNorm1d(eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.normlize2 = nn.LazyBatchNorm1d(eps=1e-6)

    def forward(self, x):   # (B, 64, 256)
        x_fft = torch.fft.fft2(x).type(torch.float32)   # (B, 64, 256)
        x = x + x_fft   # (B, 64, 256)
        x = self.normlize1(x)   # (B, 64, 256)
        x_ffn = self.ffn(x)     # (B, 64, 256)
        x = x + x_ffn           # (B, 64, 256)

        return self.normlize2(x)    # (B, 64, 256)


class gMLPLayer(nn.Module):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super(gMLPLayer, self).__init__(*args, **kwargs)
        self.channel_projection1 = nn.Sequential(
            nn.Linear
        )


class Blocks(nn.Module):
    def __init__(self, name_block_model, num_blocks):
        super(Blocks, self).__init__()
        self.num_blocks = int(num_blocks)
        if name_block_model == 'mlpmixer':
            self.block_model = MLPMixerlayer(num_patches, embedding_dim, dropout_rate)
        if name_block_model == 'fnet':
            self.block_model = FNetLayer(embedding_dim, dropout_rate)

        self.blocks = nn.ModuleList([self.block_model] * self.num_blocks)

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
        return x


blocks = Blocks('fnet', num_blocks).to(device)
model = BuildClassifier(blocks, positional_encoding=True).to(device)
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=weight_decay)


def mytrain(dataloader, model, loss_fn, optimizer):
    size = len(dataloader) * batch_size
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.type(torch.float32).to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")


def myval(dataloader, model, loss_fn):
    pass
    size = len(dataloader) * batch_size
    num_blatches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += loss_fn(pred, y).item()

            correct += torch.eq(pred.argmax(1), y).sum().item()
    val_loss /= num_blatches
    correct /= size
    print(f"Val Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return round(correct, 2)


epochs = 10
for t in range(epochs):
    print((f"Epoch {t + 1}\n---------------------------"))
    mytrain(train_dataloader, model, loss_fn, optimizer)
    myval(val_dataloader, model, loss_fn)
print("Done!")
