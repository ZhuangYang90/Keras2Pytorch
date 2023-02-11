'''
无法正常训出来，无能为力
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

# Create dataset
# Configuration parameters
POSITIVE_CLASS = 1
BAG_COUNT = 1000
VAL_BAG_COUNT = 300
BAG_SIZE = 3
PLOT_SIZE = 3
ENSEMBLE_AVG_COUNT = 1
BATCH_SIZE = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = 'D:/IMAGE/DataSet'
train_datasets = datasets.MNIST(data_path, train=True, transform=transforms.Compose([
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))]),
                                                                 download=True)
test_datasets = datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))]),
                                                                 download=True)


def create_bags(datasets, positive_class, bag_count, instance_count):  # datasets, 1, 1000, 3
    # Set up bags.
    bags = []
    bag_labels = []

    # Count positive samples.
    count = 0
    for _ in range(bag_count):
        # pytorch没有对应的np.random.choice这个方法，但有以下两个方式可以平替
        # 方法1：pictures[torch.randint(len(pictures), (10,))]
        # 方法2：indices = torch.randperm(len(pictures))[:10]  pictures[indices]

        index = torch.randperm(len(datasets))[:instance_count]
        instances_data = [datasets[i][0].tolist() for i in index]
        instances_labels = [datasets[i][1] for i in index]

        # By default, all bags are labeled as 0.
        bag_label = 0

        # Check if there is at least a positive class in the bag.
        if positive_class in instances_labels:
            # Positive bag will be labeled as 1.
            bag_label = 1
            count += 1

        bags.append(instances_data)
        bag_labels.append(bag_label)

    bags = torch.Tensor(bags).squeeze(2)
    fn = lambda x: [1, 0] if x == 0 else [0, 1]
    bag_labels = [fn(bag_label) for bag_label in bag_labels]
    bag_labels = torch.Tensor(bag_labels)

    print(f'Positive bags: {count}')
    print(f'Negative bags: {bag_count - count}')

    return bags, bag_labels


train_datasets = create_bags(train_datasets, POSITIVE_CLASS, BAG_COUNT, BAG_SIZE)
test_datasets = create_bags(test_datasets, POSITIVE_CLASS, BAG_COUNT, BAG_SIZE)
train_bags, train_labels = train_datasets
test_bags, test_labels = test_datasets


def compute_class_weights(labels):
    # Count number of postive and negative bags.
    labels = np.array(labels)
    negative_count = len(np.where(labels == 0)[0])
    positive_count = len(np.where(labels == 1)[0])
    total_count = negative_count + positive_count

    # Build class weight dictionary.
    return {
        0: (1 / negative_count) * (total_count),
        1: (1 / positive_count) * (total_count),
    }


class_weights = compute_class_weights(train_labels)


class BagDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            # 一个bag的三张不同图片分别进行transform后再组合一起
            img1 = self.transform((self.dataset[0][index][0]).unsqueeze(0))  # 考虑此处是否需要临时增加一个维度？
            img2 = self.transform((self.dataset[0][index][1]).unsqueeze(0))
            img3 = self.transform((self.dataset[0][index][2]).unsqueeze(0))
            img = torch.concat((img1, img2, img3), 0)  # 重新合成(3,28,28)
        else:
            img = self.dataset[0][index]
        label = self.dataset[1][index]
        return img, label

    def __len__(self):
        return len(self.dataset)


train_datasets = BagDataset(train_datasets)
test_datasets = BagDataset(test_datasets)

train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_datasets, batch_size=BATCH_SIZE)


# class MILAttentionLayer(nn.Module):
#     def __init__(self, input_dim, weight_params_dim, use_gated=False):
#         super(MILAttentionLayer, self).__init__()
#         self.weight_params_dim = weight_params_dim
#         self.use_gated = use_gated
#
#         self.v_weight_params = torch.empty((input_dim, self.weight_params_dim))  # input_dim = 64
#         self.w_weight_params = torch.empty((self.weight_params_dim, 1))
#         if self.use_gated:
#             self.u_weight_params = torch.empty((input_dim, self.weight_params_dim))
#         else:
#             self.u_weight_params = None
#
#         nn.init.xavier_uniform_(self.v_weight_params, gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self.w_weight_params, gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self.u_weight_params, gain=nn.init.calculate_gain('relu'))
#
#         self.w_weight_params = self.w_weight_params.to(device)
#         self.v_weight_params = self.v_weight_params.to(device)
#         self.u_weight_params = self.u_weight_params.to(device)
#
#     def compute_attention_scores(self, instance):  # (B, 64)
#         # Reserve in-case "gated mechanism" used.
#         original_instance = instance
#
#         # tanh(v*h_k^T)
#         instance = torch.mm(instance, self.v_weight_params)  # (B,64)*(64, 256) -> (B,256)
#         instance = torch.tanh(instance)  # (B, 256)
#
#         # for learning non-linear relations efficiently.
#         if self.use_gated:
#             instance = instance * torch.sigmoid(torch.mm(original_instance, self.u_weight_params))  # (B, 256)
#
#         instance = torch.mm(instance, self.w_weight_params)  # (B, 256)*(256, 1) -> (B, 1)
#         return instance
#
#     def forward(self, inputs):  # (3, B, 64)
#         instances = [self.compute_attention_scores(instance) for instance in inputs[:, ...]]
#         # [(B, 1), (B, 1), (B, 1)]
#         instances = tuple(instances)
#         instances = torch.cat(instances, dim=1)  # (B,3)
#         alpha = F.softmax(instances, dim=1)  # (B,3)
#         return alpha

class MILAttentionLayer(nn.Module):
    def __init__(self, input_dim, weight_params_dim, use_gated=False):
        super(MILAttentionLayer, self).__init__()
        self.use_gated = use_gated
        self.v_layer = nn.Sequential(
            nn.Linear(input_dim, weight_params_dim),
            nn.ReLU(),
            nn.Dropout(0.8)
        )
        self.w_layer = nn.Sequential(
            nn.Linear(weight_params_dim, 1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        if self.use_gated:
            self.u_layer = nn.Sequential(
                nn.Linear(input_dim, weight_params_dim),
                nn.ReLU()
            )
        else:
            self.u_layer = None

        nn.init.xavier_uniform_(self.v_layer[0].weight)
        nn.init.xavier_uniform_(self.w_layer[0].weight)
        nn.init.xavier_uniform_(self.u_layer[0].weight)

    def compute_attention_score(self, instance):  # (B, 64)
        # Reserve in-case "gated mechanism" used.
        original_instance = instance
        instance = self.v_layer(instance)  # (B,64)*(64, 256) -> (B,256)
        instance = torch.tanh(instance)  # (B, 256)

        # for learning non-linear relations efficiently.
        if self.use_gated:
            instance = self.u_layer(original_instance)
            instance = instance * torch.sigmoid(instance)  # (B, 256)
        instance = self.w_layer(instance)  # (B, 256)*(256, 1) -> (B, 1)
        return instance

    def forward(self, inputs):  # (3, B, 64)
        instances = [self.compute_attention_score(instance) for instance in inputs]
        # [(B, 1), (B, 1), (B, 1)]
        instances = tuple(instances)
        instances = torch.cat(instances, dim=1)  # (B,3)
        alpha = F.softmax(instances, dim=1)  # (B,3)
        return alpha


class CreateModel(nn.Module):
    def __init__(self):
        super(CreateModel, self).__init__()
        self.embeddings = []
        self.bags_multi_linear = nn.Sequential(
            nn.Flatten(start_dim=2, end_dim=-1),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.alpha_fn = MILAttentionLayer(
            input_dim=64,  # 64
            weight_params_dim=256,  # 256
            use_gated=True,  # True
        )
        self.linear = nn.Linear(192, 2)

    def forward(self, x):  # (B, 3, 28, 28)
        inp_orl = self.bags_multi_linear(x)  # (B, 3, 64)
        inp = inp_orl.permute(1, 0, 2).contiguous()  # (3, B, 64)
        alpha_out = self.alpha_fn(inp)  # (B, 3)
        alpha = alpha_out.view(-1, 3, 1)  # (B, 3, 1)
        out = torch.mul(inp_orl, alpha).view(-1, 192)  # (B, 3, 64) -> (B, 192)
        out = F.softmax(self.linear(out), dim=1)
        return out, alpha_out  # (B, 1)


model = CreateModel().to(device)
# loss_weights = torch.Tensor([class_weights[1]]).type(torch.float32).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader) * BATCH_SIZE
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.type(torch.float32).to(device), y.type(torch.float32).to(device)
        pred, _ = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")


def val(dataloader, model, loss_fn):
    size = len(dataloader) * BATCH_SIZE
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.type(torch.float32).to(device), y.type(torch.float32).to(device)
            pred, _ = model(x)
            val_loss += loss_fn(pred, y).item()

            correct += torch.eq(torch.argmax(pred), torch.argmax(y)).sum().item()
    val_loss /= num_batches
    correct /= size
    print(f"Val Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return round(correct, 2)


epochs = 1000
best_accuracy = 0
for t in range(epochs):
    print(f"Epoch {t + 1}\n---------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    val_accuracy = val(test_dataloader, model, loss_fn)
    if best_accuracy < val_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), './best_weights.pth')
    torch.save(model.state_dict(), 'last_weights.pth')
print("Done!")
