import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from my_utils.process import image_extract_pathches
import matplotlib.pyplot as plt
import numpy as np
import random


train = datasets.CIFAR100('D:/IMAGE/DataSet/CIFAR100', train=True, download=True)
val = datasets.CIFAR100('D:/IMAGE/DataSet/CIFAR100', train=False, download=True)

learning_rate = 0.001
weight_decay = 0.0001
image_size = 72
size = (image_size, image_size)
patch_size = 6
batch_size = 256
num_classes = 100
num_patches = (image_size // patch_size) ** 2
num_epochs = 10
projection_dim = 64
num_heads = 4
transformer_layers = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Use data augmentation

data_augmentation = {
    'train': transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0.02),
        transforms.RandomResizedCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class MyLazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            img = self.transform(self.dataset[index][0])
        else:
            img = self.dataset[index][0]
        label = self.dataset[index][1]
        return img, label

    def __len__(self):
        return len(self.dataset)


train_datasets = MyLazyDataset(train, transform=data_augmentation['train'])
val_datasets = MyLazyDataset(val, transform=data_augmentation['val'])
print(train_datasets[0])
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


# 随机抽取一张图片
idx = random.choice(range(len(train)))
image = train[idx][0]
# 显示原图， vision of original image
plt.figure(figsize=(4, 4))
img = np.array(image)
plt.imshow(img.astype("uint8"))
plt.axis("off")

image = transforms.ToTensor()(image)
image = transforms.Resize(size)(image).unsqueeze(0)
patches = Patches(patch_size)(image)
# print(patches.size())

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))

for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = transforms.ToPILImage()(patch)
    patch_img = np.array(patch_img)
    plt.imshow(patch_img.astype("uint8"))
    plt.axis("off")
# plt.show()


class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(108, projection_dim)
        self.position_embedding = nn.Embedding(
            num_embeddings=num_patches, embedding_dim=projection_dim
        )

    def forward(self, patch):
        positions = torch.arange(0, self.num_patches, step=1).type(torch.int64).to(device)
        patch = patch.permute(0, 2, 1).contiguous()
        pj = self.projection(patch)
        pe = self.position_embedding(positions)
        encoded = self.projection(patch) + self.position_embedding(positions)  # (256,144,64) = (256,144,64)+(144,64)
        return encoded


# Build the ViT model
class Vit_model(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(Vit_model, self).__init__()
        self.patches_fc = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.Dropout(0.1)
        )
        self.module_list = nn.ModuleList(
            [nn.LazyBatchNorm1d(1e-6),
             nn.MultiheadAttention(num_heads=num_heads, embed_dim=projection_dim, dropout=0.1),
             nn.LazyBatchNorm1d(1e-6),
             self.mlp1
             ] * transformer_layers
        )
        self.lazy_batchNorm1d = nn.LazyBatchNorm1d(1e-6)
        self.flatten = nn.Flatten()
        self.mlp2 = nn.Sequential(
            nn.Linear(9216, 2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        patches = self.patches_fc(x)
        B, PS, C, H, W = patches.size()
        patches = patches.view(B, PS * 3, H * W)  # (256, 108, 144)
        encoded_patches = self.patch_encoder(patches).permute(0, 2, 1).contiguous()
        # (256, 144, 64)->(256, 64, 144) (C, N, L)
        for i in range(*{'start': 0, 'stop': transformer_layers * 4, 'step': 4}.values()):  # 用range(0, 32, 4)这种写法会报错
            x1 = self.module_list[i](encoded_patches).permute(2, 0, 1).contiguous()  # (144, 256, 64)
            attention_output, _ = self.module_list[i + 1](x1, x1, x1)  # (144, 256, 64)
            attention_output = attention_output.permute(1, 2, 0).contiguous()
            x2 = attention_output + encoded_patches
            x3 = self.module_list[i + 2](x2)
            x3 = self.module_list[i + 3](x3.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
            encoded_patches = x3 + x2
        representation = self.lazy_batchNorm1d(encoded_patches)
        representation = self.flatten(representation)
        representation = F.dropout(representation, 0.5)
        out = self.mlp2(representation)
        return out


model = Vit_model(num_patches, projection_dim).to(device)
# print(f'model:\n{model}')
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader) * batch_size
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


def val(dataloader, model, loss_fn):
    size = len(dataloader) * batch_size
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


epochs = 10
best_accuracy = 0
for t in range(epochs):
    print(f"Epoch {t + 1}\n---------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    val_accuracy = val(val_dataloader, model, loss_fn)
    if best_accuracy < val_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), './best_weights.pth')
    torch.save(model.state_dict(), 'last_weights.pth')
print("Done!")


