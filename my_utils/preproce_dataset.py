import tqdm
import os
import shutil
from pathlib import Path
import random


def createDir_and_copyFile(path_img, subname, cls_name, files_list):
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
    return dir(path_img)