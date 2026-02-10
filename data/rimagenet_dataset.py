import json
import os

import albumentations
import numpy as np
import torch
import torchvision.transforms as trn

from collections import defaultdict
from pathlib import Path

from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from PIL import Image

import scipy.io as spio
import pickle
import scipy as sp



mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Rotation config
test_config = {}
test_config['rotations'] = np.array(range(1, 4, 2)) * 20
test_config['rotation_probs'] = np.zeros(4)
test_config['rotation_probs'][:] = 1

train_config = {}
train_config['rotations'] = np.array(range(0, 4, 2)) * 20
train_config['rotation_probs'] = np.zeros(4)
train_config['rotation_probs'][:] = 1

def read_dir(data_dir):
   
    dataset = pickle.load(open(data_dir, 'rb'))

    ans_set = dataset['labels']
    train_set = dataset['images']
    grp_set = dataset['groups']

    train_set = train_set.reshape((len(ans_set), 64, 64, 3))
    train_set = train_set.astype(np.float32)

    ans_set = ans_set.astype(np.int32)
    grp_set = grp_set.astype(np.int32)
    return list(set(grp_set)), (grp_set, train_set, ans_set)

def rotate(X, rotation, single_image=False):
    if single_image:
        return np.array(sp.ndimage.rotate(X, rotation, reshape=False, order=0))
    else:
        return np.array(
            [sp.ndimage.rotate(X[i], rotation[i], reshape=False, order=0)
             for i in range(X.shape[0])]
        )

class rimageNetDataset(Dataset):

    def __init__(self, split, root_dir):
        self.base_path = Path(root_dir) / 'Tiny-ImageNet-C'
        #if split == 'train':
            #self.root_dir = Path(root_dir) / 'rimagenet' / split / '1.pkl'
        #else:
            #self.root_dir = Path(root_dir) / 'rimagenet' / split / '1.pkl'
        #clients, data = read_dir(self.root_dir)
        
        self.corruptions = [
            'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
            'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise',
            'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise',
            'snow', 'zoom_blur'
        ]
        
        self.n_groups = len(self.corruptions)
        self.groups = list(range(self.n_groups))

        self.image_shape = (3, 64, 64)

        #agg_X, agg_y, agg_groups = [], [], []
        sample_corr = self.corruptions[0]
        class_folders = sorted([f.name for f in (self.base_path / sample_corr / '1').iterdir() if f.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_folders)}
        self.n_classes = len(class_folders)
        
        self.all_images = []
        self.all_labels = []
        self.all_groups = []
        
        print("loading rimageNet")
            
        for g_id, corr in enumerate(self.corruptions):
            for sev in ['1', '2', '3', '4', '5']:
                corr_path = self.base_path / corr / sev
            
                for cls_name in class_folders:
                    cls_path = corr_path / cls_name
                    if not cls_path.exists(): continue
                
                    label = self.class_to_idx[cls_name]
                    for img_path in cls_path.glob('*.JPEG'):
                        self.all_images.append(str(img_path))
                        self.all_labels.append(label)
                        self.all_groups.append(g_id)
        
        #for client in clients:
        #    ids = np.nonzero(data[0] == client)[0]
        #   client_X = data[1][ids]
        #    client_y = data[2][ids]
        #    client_N = len(client_X)
            
        #    X_processed = np.array(client_X).reshape((client_N, 64, 64, 3))

        #    agg_X.append(X_processed)
        #    agg_y.extend(client_y)
        #    agg_groups.extend([self.groups.index(client)] * client_N)
#             agg_groups.extend([client] * client_N)

        print("loaded")
        #self._len = len(agg_groups)
        #self._X, self._y = np.concatenate(agg_X), np.array(agg_y)
        self.group_ids = np.array(self.all_groups)
        
        #self.original_size = len(self._X)
        
        #self.all_indices = range(self.original_size)
        
        #if split == 'train':
        #    skew_config = train_config
        #    self.indices, self.rotations, self.rotation_ids = self._get_train_skew(skew_config)
        #else:
        #    skew_config = test_config
        #    self.indices, self.rotations, self.rotation_ids = self._get_test(skew_config)
               
        self._y = np.array(self.all_labels)
        #self._X, self._y = self._X[self.indices], self._y[self.indices]
        #self.group_ids = self.group_ids[self.indices]
        #self._get_group_ids(self.rotation_ids)
        self._len = len(self.all_images)
        
        #self.groups = list(set(self.group_ids))
        #self.n_groups = len(self.groups)
        
        self.group_counts, _ = np.histogram(self.group_ids,
                                            bins=range(self.n_groups + 1)
                                            )#density=False)
        self.group_dist, _ = np.histogram(self.group_ids, 
                                          bins=range(self.n_groups+1), 
                                          density=True)
        
        #self.group_dim = 2
        #self.group_id_processor = {1: self._dist_getter,
        #                           2: self._rot_getter}
        #self.subgroups = {1: np.asarray([self._dist_getter(x) for x in self.groups]), 
        #                  2: np.asarray([self._rot_getter(x) for x in self.groups])}
        
        
        # Classes
        self.n_classes = 200
        self.classes = np.array(range(self.n_classes))
#         self.classes = np.asarray(sorted(list(set(self._y))))
        self.class_with_ids = self._get_class_ids(self._y)
        
        self.transform = self.get_transform()

        print(f"Loaded {self._len} images from {self.n_groups} corruption groups.")
        #print("split: ", split)
        #print("n groups: ", self.n_groups)
        #print("Dataset size: ", len(self._y))
        
        self.rotation_ids = np.zeros(len(self.all_images))

        print("Smallest group: ", np.min(self.group_counts))
        print("Largest group: ", np.max(self.group_counts))

    def __len__(self):
        return self._len

    def _get_test(self, skew_config):
        rotations = []
        rotation_ids = []
        indices = []

        for rotation_id, rotation in enumerate(skew_config['rotations']):

            rotations.extend([rotation] * self.original_size)
            rotation_ids.extend([rotation_id] * self.original_size)
            indices.extend(self.all_indices)

        rotations = np.array(rotations)
        rotation_ids = np.array(rotation_ids)
        indices = np.array(indices)

        return indices, rotations, rotation_ids

    def _get_train_skew(self, skew_config):
        """Returns a skewed train set"""

        num_examples_total = len(self._y)

        indices = []
        rotations = []
        rotation_ids = []
        for rotation_id, rotation in enumerate(skew_config['rotations']):
            rotation_prob = skew_config['rotation_probs'][rotation_id]
            group_prob = rotation_prob
#             num_examples = int(rotation_prob * self.original_size)
#             indices_for_rotation = np.random.choice(self.original_size, size=num_examples)
            indices_for_rotation = self.all_indices
            rotations.append(len(indices_for_rotation) * [rotation])
            rotation_ids.append(len(indices_for_rotation) * [rotation_id])
            indices.append(indices_for_rotation)

        rotations = np.concatenate(rotations)
        rotation_ids = np.concatenate(rotation_ids)
        indices = np.concatenate(indices)

        return indices, rotations, rotation_ids
    
    def _get_class_ids(self, labels):
        """Returns the class ids for each example

            TODO: Clean up this function"""

        class_with_ids = {}
        for cls in self.classes:
            ids = np.nonzero(labels == cls)[0]
            
            if len(ids) > 0:
                class_with_ids[cls] = ids
            else:
                print(f"Warning: Class {cls} is empty in this split!")

        return class_with_ids
    
    def _get_group_ids(self, rotation_ids):
        for idx, rotation_id in enumerate(rotation_ids):
            self.group_ids[idx] = self._id_getter(self.group_ids[idx], rotation_id)

    def _id_getter(self, gid, rotation_id):
        return gid*2+rotation_id
    
    def _dist_getter(self, gid):
        return int(gid/2)
    
    def _rot_getter(self, gid):
        return int(gid%2)
        

    def __getitem__(self, index):
        img = Image.open(self.all_images[index]).convert('RGB')
        img = np.array(img)
        
        #rotation_index = index
        #index = index % self.original_size
        
        #x = self._X[index]
        #rotation = self.rotations[rotation_index]
        #x = rotate(x, rotation, single_image=True)
            
        x = self.transform(image=img)['image']
        y = torch.tensor(self._y[index], dtype=torch.long)
        g = torch.tensor(self.group_ids[index], dtype=torch.long)
        return x, y, g

    def get_transform(self):

        return albumentations.Compose([
                                    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                             #max_pixel_value=255, p=1.0, always_apply=True),
                                    ToTensor()])

       

        #return transform
