import random

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F

from my_utils.preproce_dataset import split_image_folder
from my_utils.model_cell import SeparableConv2d

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sourse', type=str, default="F:/DataSet/IMAGES/kagglecatsanddogs_5340/PetImages",
                        help='files_path for train and val')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[180, 180],
                        help='inference size h,w')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epoch', type=int, default=1, help='times of the training loop.')
    parser.add_argument('--val_split', type=float, default=0.2, help='rate of val')
    parser.add_argument('--test_split', type=float, default=0.1, help='rate of test')
    parser.add_argument('--retrain', type=bool, default=True, help='whether to train with pretraining weigths')
    parser.add_argument('--kinds', type=list, default=['Cat', 'Dog'], help='')
    opt = parser.parse_args()

    return opt


opt = parse_opt()

'''
在keras示例中已经对数据进行处理，此处就不再重复
'''

# Generate a Dataset
# 图片所在文件夹
path_img, val_split, test_split, image_size, epochs, batch_size, retrain, kinds \
    = opt.sourse, opt.val_split, opt.test_split, opt.imgsz, opt.epoch, opt.batch_size, opt.retrain, opt.kinds

# 一个function将数据集修改成带有train和val以及test存放方式格式如下，比例设定成（7，2，1），从而可以直接套用ImageFolder方法
'''
Folder:
    train:
        class_a:
            xx.png...
        class_b:
            xxx_png...
    val:
        class_a:
            xy.png...
        class_b:
            xxy_png...
'''
'''def createDir_and_copyFile(path_img, subname, cls_name, files_list):
    path_img = str(path_img)
    sub_dir = os.path.join(path_img, subname)
    target_dir = os.path.join(sub_dir, cls_name)
    os.makedirs(target_dir, exist_ok=True)
    for file in files_list:
        name = file.name
        target_file = target_dir+os.sep+name
        shutil.copyfile(file, target_file)
    return


def split_image_folder(path_img, val_split, test_split):
    rate_val, rate_test = val_split, test_split
    rate_train = 1 - rate_val - rate_test
    path_img = Path(path_img)
    classes_list = [sub_dir for sub_dir in path_img.iterdir()]  # [path_'Cat', path_'Dog']

    for cls in classes_list:

        cls_name = cls.name

        img_list = [file for file in cls.glob('*.jpg')]
        random.shuffle(img_list)
        num_files = len(img_list)
        train_list = img_list[:int(num_files*rate_train)]
        val_list = img_list[int(num_files*rate_train):int(num_files*(rate_train+rate_val))]
        test_list = img_list[-1*int(num_files*rate_test):]

        for subname in tqdm(('train', 'val', 'test')):
            if subname == 'train':
                files_list = train_list
            elif subname == 'val':
                files_list = val_list
            elif subname == 'test':
                files_list = test_list
            createDir_and_copyFile(path_img, subname, cls_name, files_list)

        # 删除原先的数据集,慎用
        # os.rmdir(cls)
    return dir(path_img)'''

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0.1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
    'val': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataset_folder = path_img
image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_folder, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
               for x in ['train', 'val']}
dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Build a model
class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, kernel_size, bias),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(out_channels, out_channels, kernel_size, bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        outputs = F.max_pool2d(x, 1, 2)
        return outputs


class Make_model(nn.Module):
    def __init__(self, num_classes):
        super(Make_model, self).__init__()

        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block1 = bottleneck(128, 256, 3)
        self.block2 = bottleneck(256, 512, 3)
        self.block3 = bottleneck(512, 728, 3)

        self.res_conv1 = nn.Conv2d(128, 256, 1, 2)
        self.res_conv2 = nn.Conv2d(256, 512, 1, 2)
        self.res_conv3 = nn.Conv2d(512, 728, 1, 2)

        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(1024, 1)
        self.linear_n = nn.Linear(1024, self.num_classes)

        self.conv2 = nn.Sequential(
            SeparableConv2d(728, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))
        )

    @autocast()
    def forward(self, x):
        x = torch.div(x, 255)
        x = self.conv1(x)  # (4, 128, 90, 90)
        previous_block_activation = x  # Set aside residual     # (90,90)

        x = self.block1(x)
        residual = self.res_conv1(previous_block_activation)
        x += residual
        previous_block_activation = x

        x = self.block2(x)
        residual = self.res_conv2(previous_block_activation)
        x += residual
        previous_block_activation = x

        x = self.block3(x)
        residual = self.res_conv3(previous_block_activation)
        x += residual

        x = self.conv2(x)
        x = F.dropout2d(x, 0.5)
        x = self.flatten(x)
        if self.num_classes == 2:
            x = self.linear_1(x)
            outputs = torch.sigmoid(x)
        else:
            x = self.linear_n(x)
            outputs = F.softmax(x)
        return outputs


def mytrain(dataloader, datasize, model, loss_fn, optimizer):
    size = datasize
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device).type(torch.float16)
        # y = y.unsqueeze(1)

        pred = model(x).squeeze(1)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")


def myval(dataloader, datasize, model, loss_fn):
    size = datasize
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct, best_correct = 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device).type(torch.float16)
            pred = model(x).squeeze(1)
            val_loss += loss_fn(pred, y).item()
            pred = torch.ge(pred, 0.5).type(torch.float16)
            correct += torch.eq(pred, y).sum().item()
    val_loss /= num_batches
    correct /= size
    print(f"Val Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return round(correct, 2)


model = Make_model(2).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def run(epochs):
    train_dataloader = dataloaders['train']
    train_size = dataset_size['train']
    val_dataloader = dataloaders['val']
    val_size = dataset_size['val']
    if retrain:
        model.load_state_dict(torch.load('./model_weights.pth'))
    accuracy = 0
    for t in range(epochs):
        print(f'Epoch{t + 1}\n-------------------------------')
        mytrain(train_dataloader, train_size, model, loss_fn, optimizer)
        val_accuracy = myval(val_dataloader, val_size, model, loss_fn)
        if accuracy < val_accuracy:
            accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_weights.pth')
    torch.save(model.state_dict(), 'last_weights.pth')


# vision of the test
def vision(test_dataset, title):
    model = Make_model(2).to(device)
    model.load_state_dict(torch.load('./best_weights.pth'))
    plt.figure(figsize=(10, 10)).suptitle(title, fontsize=18)
    for i, img in enumerate(test_dataset[:9]):
        img = Image.open(img)
        img = img.resize(tuple(image_size), Image.BILINEAR)
        img = np.array(img)
        im = torch.from_numpy(img).to(device)
        pred = model(im)
        idx = int(torch.ge(pred, 0.5).item())
        cls_name = kinds[idx]
        plt.subplot(3, 3, i + 1)
        plt.imshow(img.astype("uint8"))
        plt.title(cls_name)
        plt.axis("off")
    plt.show()


if __name__ == '__main__':
    # dataset_folder = split_image_folder(path_img, val_split, test_split)     # 已经进行处理
    # run(epochs)

    path_test = os.path.join(path_img, 'test')
    path_test = Path(path_test)
    samples_list = [img for img in path_test.glob('*.jpg')]
    random.shuffle(samples_list)
    samples_list = samples_list[:9]
    vision(samples_list, title='Test of Cat & Dog')
