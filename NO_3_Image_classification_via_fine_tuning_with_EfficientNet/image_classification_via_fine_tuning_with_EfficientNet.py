import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet18

from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 512
IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)
NUM_CLASSES = 120
rate_for_training = 0.8
rate_for_val = 0.1
untrainable_layers_num = [6]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def vision_data_balance_distribution(samples_list):
    samples_dict = Counter(samples_list)
    keys = [elem for elem in samples_dict.keys()]
    values = [samples_dict[k] for k in keys]
    sns.barplot(x=keys, y=values)
    plt.show()


# Data preparation
path_img = 'F:/DataSet/IMAGES/stanford_dogs/images'
path_img = Path(path_img)
# 种类的名称，按照顺序排，因为ImageFolder中的classes (list): List of the class names sorted alphabetically.
class_names = [subdir.name for subdir in path_img.iterdir()]
class_names.sort()
# 统计出一共有多少个数据
imgs_list = list(iter(path_img.glob('*/*.jpg')))
num_imgs = len(imgs_list)
# print(f'num_imgs:{num_imgs}')

num_train = int(rate_for_training * num_imgs)
num_val = int(rate_for_val * num_imgs)
num_test = num_imgs - num_train - num_val
# print("number for training:{}, number for val:{}, number for test:{}".format(num_train, num_val, num_test))
'''
delete the file without JFIF,数据已经做过这一步就不赘述
'''

# Data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0.1),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
    'val': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# build datasets
'''
以下这个方法可以尝试，这样做就可以不用再去将源数据集进行改动了

dataset=torchvision.datasets.ImageFolder('path')
train, val, test = torch.utils.data.random_split(dataset, [1009, 250, 250])
traindataset = MyLazyDataset(train,aug)
valdataset = MyLazyDataset(val,aug)
testdataset = MyLazyDataset(test,aug)
num_workers=2
batch_size=6
trainLoader = DataLoader(traindataset , batch_size=batch_size, 
                                           num_workers=num_workers,  shuffle=True)
valLoader = DataLoader(valdataset, batch_size=batch_size, 
                                          num_workers=num_workers )
testLoader = DataLoader(testdataset, batch_size=batch_size, 
                                          num_workers=num_workers)

该方法来自：
https://discuss.pytorch.org/t/using-imagefolder-random-split-with-multiple-transforms/79899/7
'''
dataset = datasets.ImageFolder(path_img)
# check_balance_all = [lb for _, lb in dataset]
# print("Check the balance of the total datas:")
# vision_data_balance_distribution(check_balance_all)
train, val, test = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])


# # 以下用来检测上面这种方法数据分布的平衡情况
# check_balance = [lb for _, lb in test]
# print("Check the balance of the test datas:")
# vision_data_balance_distribution(check_balance)   # 结果是，这种方法分配不算平均但是能用

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
        return len(dataset)


train_datasets = MyLazyDataset(train, data_transforms['train'])
val_datasets = MyLazyDataset(val, data_transforms['val'])
test_datasets = MyLazyDataset(test, None)

train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Build a model
weights = EfficientNet_B0_Weights.DEFAULT
pre_model = efficientnet_b0(weights=weights)
# print(list(iter(pre_model.named_modules())))    # 查看模型的结构
# print('--------------------------------')
# print(list(iter(pre_model.named_parameters())))   # 确定冻结其前6层
# print('-----------------------------')
# 替换模型最后的输出层，将层数改成120个
# last_layer = pre_model.classifier     # 查看最后一层结构，方便后面重新构造
# print(f'num_feature:{last_layer}')
new_last_layer = nn.Sequential(
    nn.Dropout(0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=120, bias=True)
)
pre_model.classifier = new_last_layer



class Build_model(nn.Module):
    def __init__(self):
        super(Build_model, self).__init__()
        self.model = pre_model

    def forward(self, x):
        out = self.model(x)
        return out


model = Build_model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def mytrain(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)*BATCH_SIZE
    # 设定可以训练的层数，冻结前6层，训练后两层
    freeze = untrainable_layers_num
    freeze = [f'features.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in pre_model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            v.requires_grad = False
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.type(torch.float32).to(device), y.to(device)

        pred = model(x).squeeze(1)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")


def myval(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    size = num_batches*BATCH_SIZE
    model.eval()
    val_loss, correct= 0, 0
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


def train_val_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, weights=None):
    size = len(train_dataloader)*BATCH_SIZE
    if isinstance(weights, str):
        model.load_state_dict(torch.load(weights))
    # 设定可以训练的层数，冻结前6层，训练后两层
    freeze = untrainable_layers_num
    freeze = [f'features.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in pre_model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            v.requires_grad = False
    accuracy = 0

    for t in range(epochs):
        print(f"Epoch {t + 1}\n---------------------------")
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.type(torch.float32).to(device), y.to(device)

            pred = model(x).squeeze(1)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")

        val_accuracy = myval(val_dataloader, model, loss_fn)
        if accuracy < val_accuracy:
            accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_weights.pth')
        torch.save(model.state_dict(), 'last_weights.pth')


if __name__ == '__main__':      # 含有freeze需要在main中才能运行
    '''    
    # 以下这种写法，每次都需要冻结后再训练，因此速度会很慢，将epochs循环直接放到train函数当中减少不必要的步骤
    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------------")
        mytrain(train_dataloader, model, loss_fn, optimizer)
        myval(val_dataloader, model, loss_fn)
    print("Done!")
    '''
    epochs = 5
    train_val_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs)
