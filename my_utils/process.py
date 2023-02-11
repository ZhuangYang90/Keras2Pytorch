import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt


def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2))

    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()

    return patches.view(b, -1, patches.shape[-2], patches.shape[-1])


def image_extract_pathches(x, kernel_size=3, dilation=1, stride=3):
    x = x.float()
    x = F.unfold(x, kernel_size=kernel_size, dilation=dilation, stride=stride)
    B, C_kh_kw, L = x.size()
    x = x.permute(0, 2, 1).contiguous()
    x = x.view(B, L, -1, kernel_size, kernel_size)
    x = x.squeeze(2)
    return x


def test():
    # input = torch.arange(9 * 9).view(1, 1, 9, 9)
    # print(input)
    # print(input.size())
    # print()
    # output = extract_image_patches(input, 3, stride=3)
    # n_output = output.view(1, 1, 9, 9).permute(0, 1, 3, 2).contiguous()
    #
    # print(n_output)

    # x = torch.arange(0, 1 * 1 * 16 * 16).float()
    # x = x.view(1, 1, 16, 16)
    # print(x)
    # x = image_extract_pathches(x, 4, 1, 4)
    # x = x.squeeze(2)
    # print(x.size())
    img = 'C:/DataSet/IMAGES/kagglecatsanddogs_5340/PetImages/Cat/1.jpg'
    img = Image.open(img)
    img_t = transforms.ToTensor()(img)
    img_fft = torch.fft.fft2(img_t).type(torch.float32)
    img_t = img_fft + img_t
    # print(img_t.size())
    img_t = transforms.ToPILImage()(img_t)

    plt.figure(figsize=(5,10))
    ax = plt.subplot(1,2,1)
    plt.imshow(img)
    ax = plt.subplot(1,2,2)
    plt.imshow(img_t)
    plt.show()

test()




