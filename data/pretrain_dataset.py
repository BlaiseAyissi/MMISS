import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption
import os,glob

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, laion_path, transform): 
            
        self.transform = transform 
        
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]   
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
        caption = pre_caption(ann['caption'],30)
        
        return image, caption