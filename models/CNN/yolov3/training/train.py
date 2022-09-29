import torch
import dataloader
import model
import loss
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from clearml import Task
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
c = 0
import warnings
warnings.filterwarnings("ignore")

task = Task.init(project_name='Object Detection', task_name='mscoco testing  with additional gt removed seperated predictions OVERFIT')
def test_fn(train_loader, model, epoch, optimizer, scheduler, loss_fn, scaler):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loop = tqdm(train_loader, leave=True)
    losses = []
    #model.train()
    
    for batch_idx, (x, y1, y2) in enumerate(loop):
        x = x.to(device)
        y10, y11, y12 = (
            y1[0].to(device),
            y1[1].to(device),
            y1[2].to(device),
        )
        y20, y21, y22 = (
            y2[0].to(device),
            y2[1].to(device),
            y2[2].to(device),
        )
        model = model.to(device)
        #print( x.device)
        with torch.cuda.amp.autocast():
            out = model(x)
            loss1,_ = loss_fn(out[0], out[1], y10, y20)  
            loss2, _ = loss_fn(out[2], out[3], y11, y21)
            loss3, _ = loss_fn(out[4], out[5], y12, y22)
            loss = (loss1 + loss2 + loss3)/3
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    return mean_loss

def train_fn(train_loader, model, epoch, optimizer, scheduler, loss_fn, scaler):
    global c
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loop = tqdm(train_loader, leave=True)
    losses = []
    model.train()
    
    for batch_idx, (x, y1, y2) in enumerate(loop):
        x = x.to(device)
        y10, y11, y12 = (
            y1[0].to(device),
            y1[1].to(device),
            y1[2].to(device),
        )
        y20, y21, y22 = (
            y2[0].to(device),
            y2[1].to(device),
            y2[2].to(device),
        )
        model = model.to(device)
        #print( x.device)
        with torch.cuda.amp.autocast():
            out = model(x)
            loss1, tmp1 = loss_fn(out[0], out[1], y10, y20)  
            loss2, tmp2 = loss_fn(out[2], out[3], y11, y21)
            loss3, tmp3 = loss_fn(out[4], out[5], y12, y22)
            loss = (loss1 + loss2 + loss3)/3
            tmp = tmp1 + tmp2 + tmp3
            writer.add_scalar('Loss/no_obj', tmp[0], c)
            writer.add_scalar('Loss/obj', tmp[1], c)
            writer.add_scalar('Loss/class', tmp[2], c)
            writer.add_scalar('Loss/box', tmp[3], c)
            writer.add_scalar('Loss', loss, c)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], c)
            c+=1
        
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        my_lr = scheduler.optimizer.param_groups[0]['lr']
        loop.set_postfix(loss=loss.item(), epoch=epoch,  lr=my_lr )
    #scheduler.step(mean_loss)
    return mean_loss




def main():
    yolo = model.ObjectDetectionModel()
    #model_dict = OrderedDict()
    #for k, v in torch.load("./no_Sep_29.120826721191406.pt").items():
    #    model_dict[k[7:]] = v
    #yolo.load_state_dict(model_dict)
    yolo = nn.DataParallel(yolo, device_ids = [0,1,2,3])
    optimizer = optim.Adam(
        yolo.parameters(), lr=0.001, weight_decay=1e-4
    )
    loss_fn = loss.LossFn()
    scaler = torch.cuda.amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5)
    train_data = dataloader.OpenImagesDataLoader("./mscoco_train.json", "./labels.json", "./refined_labels.json", s=[16, 32, 64], length=40)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=40,
        num_workers=4,
        shuffle=True,
      collate_fn=dataloader.collate_fn
    )
    
    test_data = dataloader.OpenImagesDataLoader("./mscoc_test.json", "./labels.json", "./refined_labels.json", s=[16, 32, 64], length=1000)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=20,
        num_workers=4,
        shuffle=True,
      collate_fn=dataloader.collate_fn
    )
    NUM_EPOCHS = 500
    previous_loss = 99999
    for epoch in range(NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        mean_loss = train_fn(train_loader, yolo, epoch, optimizer, scheduler, loss_fn, scaler)
        scheduler.step(mean_loss)


        if  epoch % 10 == 0  :
            torch.save(yolo.state_dict(), "./overfit.pt")
            
        #yolo.eval()
        #curr_loss = test_fn(test_loader, yolo, epoch, optimizer, scheduler, loss_fn, scaler)            
        #if previous_loss > curr_loss:
        #    torch.save(yolo.state_dict(), "./no_Sep_"+str(curr_loss)+".pt")
        #    previous_loss = curr_loss

        #if epoch == NUM_EPOCHS -1:
        #    torch.save(yolo.state_dict(), "./no_Sep_last_"+str(curr_loss)+".pt")

        yolo.train()
                
main()

