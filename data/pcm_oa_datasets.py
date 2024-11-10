import os
import torch
import numpy as np
import random
import pandas as pd
import torch.utils.data as Data
import matplotlib as plt
import matplotlib.image as mpig
import json
#import cv2

from scipy import ndimage
from scipy.ndimage import zoom
from PIL import Image
from torchvision import transforms
from data.utils import pre_caption


class Pcm_oa_dataset(Data.Dataset):
    def __init__(self, data_dir, json_desc,transform):        # data_dir和label_dir分别表示图像和标签路径
        
        self.sample_list = []
        self.label_csv = []
        with open(json_desc) as f1:
            for line in f1:
                j_line=json.loads(line)
                self.sample_list.append(j_line['image'])
                #self.label_csv.append(pre_caption(j_line['caption'],40))
                self.label_csv.append(j_line['caption'])

        #self.list_dir = list_dir    # 索引目录路径  list of the images
        #self.sample_list = open(list_dir).readlines() 
        #self.label_csv = caption_csv  #captions  # 采样图片名称
        
        self.data_dir = data_dir  #images 
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')      # 按值索引到图片名称
        data_path = os.path.join(self.data_dir, slice_name)
        data = Image.open(data_path)   # 找到图片对应的png文件
        if self.transform:
            data = self.transform(data)
            data = data/255
        label_path = slice_name     # 按标签索引.csv文件的某行
        caption = self.label_csv[idx]  # 找到图片对应标签
        return data, str(caption)
    
