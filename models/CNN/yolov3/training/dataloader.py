from locale import currency
import os
import cv2
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def display(img, fig=(20, 20)):
    plt.figure(figsize=fig)
    plt.imshow(img)
    plt.show()

def convert_xyxy_xywhxyxy(boxes):
    #print(boxes)
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]
    c = boxes[..., 4]
    xc = (x1 + x2)/2
    yc = (y1 + y2)/2
    w = torch.abs(x2 - x1)
    h = torch.abs(y2 - y1)
    xc = xc.reshape(-1, 1)
    yc = yc.reshape(-1, 1)
    w = w.reshape(-1, 1)
    h = h.reshape(-1, 1)
    c = c.reshape(-1, 1)
    x1 = x1.reshape(-1, 1)
    y1 = y1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)
    out = torch.cat([xc, yc, w, h, x1, y1, x2, y2, c], dim=-1)
    return out


def normalize_xywhxyxy(boxes, img_shape):
    x = boxes[..., 0]
    y = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    x1 = boxes[..., 4]
    y1 = boxes[..., 5]
    x2 = boxes[..., 6]
    y2 = boxes[..., 7]
    c = boxes[...,8]
    x = x/512
    y = y/512
    w = w/512
    h = h/512
    x1 = x1/512
    y1 = y1/512
    x2 = x2/512
    y2 = y2/512
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    h = h.reshape(-1, 1)
    x1 = x1.reshape(-1, 1)
    y1 = y1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)
    c = c.reshape(-1, 1)
    out = torch.cat([x, y, w, h, x1, y1, x2, y2, c], dim=-1)
    return out

def collate_fn(batch):
    count = 0
    for _ in batch:
        count += 1 
    img_batch = np.zeros((count, 3, 512, 512), dtype=np.float16) 
    count = 0
    ctmp1, ctmp2, ctmp3 = [], [], []
    btmp1, btmp2, btmp3 = [], [], []
    for img, tar1, tar2 in batch:
        img_batch[count, :, :img.shape[1], :img.shape[2]] = img
        ctmp1.append(tar1[0])
        ctmp2.append(tar1[1])
        ctmp3.append(tar1[2])

        btmp1.append(tar2[0])
        btmp2.append(tar2[1])
        btmp3.append(tar2[2])

        count += 1
    ctmp1 = torch.from_numpy(np.array(ctmp1)).float()
    ctmp2 = torch.from_numpy(np.array(ctmp2)).float()
    ctmp3 = torch.from_numpy(np.array(ctmp3)).float()

    btmp1 = torch.from_numpy(np.array(btmp1)).float()
    btmp2 = torch.from_numpy(np.array(btmp2)).float()
    btmp3 = torch.from_numpy(np.array(btmp3)).float()

    tar1 = (ctmp1, ctmp2, ctmp3)
    tar2 = (btmp1, btmp2, btmp3)
    return torch.from_numpy(img_batch).float(), tar1, tar2


def image_resize(img, max_shape):
    aspect = img.shape[1] / img.shape[0]
    new_shape = [max_shape, int(max_shape * aspect)] if np.argmax(img.shape[:2]) == 0 else [int(max_shape / aspect), max_shape]
    img = cv2.resize(img, tuple(new_shape[::-1]), cv2.INTER_AREA)
    return img


class OpenImagesDataLoader(Dataset):


    def __init__(self, path, lab, ref, s=[16,32,64], length=-1):
        with open(path, "r") as f:
            self.paths = json.load(f)
        self.jsonfile = list(self.paths.keys())
        with open(lab, "r") as f:
            self.labels = json.load(f)
        with open(ref, "r") as f:
            self.ref_labels = json.load(f)
        self.jsonfile = list(self.paths.keys())
        self.jsonfile = self.jsonfile[:length]
        self.s = s

    def __len__(self):
        return len(self.jsonfile)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[self.jsonfile[idx]])
        img = image_resize(img, 512)
        with open(self.jsonfile[idx], "r") as f:
            data = json.load(f)
        img = (img - 127)/128
        new_box = []
        for k in data:
            if k != "img":
                b = data[k]["bbox"]
                lab = data[k]["labels"].replace("/m/", "")
                new_box.append(list(map(float, [b[0], b[1], b[2], b[3], self.ref_labels[lab]['idx']] )))

        for i in range(len(new_box)):
            new_box[i] = [new_box[i][0]*img.shape[1], new_box[i][1]*img.shape[0], new_box[i][2]*img.shape[1], new_box[i][3]*img.shape[0], new_box[i][4]]
        #print("file - ",new_box)
        
        boxes = torch.from_numpy(np.array(new_box))
        boxes = convert_xyxy_xywhxyxy(boxes)
        boxes = normalize_xywhxyxy(boxes, img.shape)
        #print("conv - ",boxes)
        boxes = boxes.detach().numpy()
        
        class_targets = [np.zeros((s, s, 1), dtype=np.float) for s in self.s]
        box_targets = [np.zeros((s, s, 5), dtype=np.float) for s in self.s]
        for box in boxes:
            x,y,w,h,xmin,ymin,xmax,ymax,c = box
            for st in range(len(self.s)):
                curr_stride = self.s[st]
                j,i = int(curr_stride * x), int(curr_stride * y)
                box_targets[st][i, j, 0] = 1
                x1, y1 = curr_stride * x - j, curr_stride * y - i 
                w1, h1 = curr_stride * w, curr_stride * h 
                box_targets[st][i, j, 1:5] = [x1, y1, w1, h1]
                class_targets[st][i, j, 0] = c
        img = np.moveaxis(img, -1, 0)    
        return img, class_targets, box_targets


class OpenImagesDataLoaderUpdatedYolo(Dataset):

    def __init__(self, path, lab, ref, s=[16,32,64], length=-1):
        with open(path, "r") as f:
            self.paths = json.load(f)
        self.jsonfile = list(self.paths.keys())
        with open(lab, "r") as f:
            self.labels = json.load(f)
        with open(ref, "r") as f:
            self.ref_labels = json.load(f)
        self.jsonfile = list(self.paths.keys())
        self.jsonfile = self.jsonfile[:length]
        self.s = s

    def __len__(self):
        return len(self.jsonfile)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[self.jsonfile[idx]])
        img = image_resize(img, 512)
        with open(self.jsonfile[idx], "r") as f:
            data = json.load(f)
        img = (img - 127)/128
        new_box = []
        for k in data:
            if k != "img":
                b = data[k]["bbox"]
                lab = data[k]["labels"].replace("/m/", "")
                new_box.append(list(map(float, [b[0], b[1], b[2], b[3], self.ref_labels[lab]['idx']] )))

        for i in range(len(new_box)):
            new_box[i] = [new_box[i][0]*img.shape[1], new_box[i][1]*img.shape[0], new_box[i][2]*img.shape[1], new_box[i][3]*img.shape[0], new_box[i][4]]
        #print("file - ",new_box)
        
        boxes = torch.from_numpy(np.array(new_box))
        boxes = convert_xyxy_xywhxyxy(boxes)
        boxes = normalize_xywhxyxy(boxes, img.shape)
        #print("conv - ",boxes)
        boxes = boxes.detach().numpy()
        boxes = sorted(boxes, reverse=True, key=lambda k :k[2]*k[3])
        ## sort and add all coords 

        ##
       
        
        class_targets = [np.zeros((s, s, 1), dtype=np.float) for s in self.s]
        box_targets = [np.zeros((s, s, 5), dtype=np.float) for s in self.s]
        for box in boxes:
            x,y,w,h,xmin,ymin,xmax,ymax,c = box
            for st in range(len(self.s)):
                curr_stride = self.s[st]
                ## center
                j,i = int(curr_stride * xmin), int(curr_stride * ymin)
                j1,i1 = int(curr_stride * xmax), int(curr_stride * ymax)
                box_targets[st][i:i1, j:j1, 0] = 1
                x1, y1 = curr_stride * x - j, curr_stride * y - i 
                w1, h1 = curr_stride * w, curr_stride * h 
                #print(xmin, ymin, xmax, ymax)
                box_targets[st][i:i1, j:j1, 1:5] = [xmin, ymin, xmax, ymax]
                class_targets[st][i:i1, j:j1, 0] = c

                
        img = np.moveaxis(img, -1, 0)    
        return img, class_targets, box_targets
    
    
import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)
