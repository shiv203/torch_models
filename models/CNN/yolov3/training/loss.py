### losses .... 
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def ioufn(pred, target):
    
    b1_x1 = pred[...,0:1]
    b1_y1 = pred[...,1:2]
    b1_x2 = pred[...,2:3]
    b1_y2 = pred[...,3:4]
    
    b2_x1 = pred[...,0:1]
    b2_y1 = pred[...,1:2]
    b2_x2 = pred[...,2:3]
    b2_y2 = pred[...,3:4]
    
    x1 = torch.max(b1_x1, b2_x2)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)
    
    inter = (x2 - x2).clamp(0) * (y2 - y1).clamp(0)
    
    b1 = torch.abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b2 = torch.abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
    
    iou = inter/(b1 + b2 - inter + 1e-6)
    return iou

class LossFnCoord(nn.Module):

    def __init__(self):
        super(LossFnCoord, self).__init__()
        self.mse = nn.MSELoss()
        self.ce = FocalLoss()
        self.sig = nn.Sigmoid()
        self.bce = nn.BCEWithLogitsLoss()
        self.no_obj_w = 1
        self.obj_w = 1
        self.class_w = 1
        self.box_w = 10
        #self.iou = 10

    def forward(self, pred_class, pred_box, gt_class, gt_box):
        #print(pred_class.shape, pred_box.shape, gt_class.shape, gt_box.shape)
        obj_present = gt_box[..., 0] == 1
        obj_absent = gt_box[..., 0] == 0

        ## no obj loss
        no_obj_loss = self.bce(pred_box[...,0:1][obj_absent], gt_box[...,0:1][obj_absent])

        ## obj loss
        obj_loss = self.bce(pred_box[...,0:1][obj_present], gt_box[...,0:1][obj_present])

        ## class loss
        class_loss = self.ce(pred_class[obj_present], gt_class[...,0][obj_present].long())

        ## box loss
        #pred_box[...,1:5] = self.sig(pred_box[...,1:5])
        pred_box[...,1:5] = self.sig(pred_box[...,1:5])
        box_loss = self.mse(pred_box[...,1:5][obj_present], gt_box[...,1:5][obj_present])

        ## iou loss
        #iou = 1 - torch.mean(ioufn(pred_box[...,1:5][obj_present], gt_box[...,1:5][obj_present]))

        print(no_obj_loss.item(), obj_loss.item(), class_loss.item(), box_loss.item())
        return self.no_obj_w * no_obj_loss + self.obj_w * obj_loss + self.class_w * class_loss + self.box_w * box_loss #+ self.iou * iou
        
class LossFn(nn.Module):


    def __init__(self):
        super(LossFn, self).__init__()
        self.mse = nn.MSELoss()
        self.ce = FocalLoss()
        self.sig = nn.Sigmoid()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()
        self.no_obj_w = 10
        self.obj_w = 1
        self.class_w = 1
        self.box_w = 10
       

    def forward(self, pred_class, pred_box, gt_class, gt_box):
        #print(pred_class.shape, pred_box.shape, gt_class.shape, gt_box.shape)
        obj_present = gt_box[..., 0] == 1
        obj_absent = gt_box[..., 0] == 0

        #pred1 = pred_box[...,0:1].clone()
        #gt1 =  gt_box[...,0:1].clone()
        #focal_loss = self.focal(pred1, gt1)

        ## no obj loss
        no_obj_loss = self.bce(pred_box[...,0:1][obj_absent], gt_box[...,0:1][obj_absent])

        ## obj loss
        obj_loss = self.bce(pred_box[...,0:1][obj_present], gt_box[...,0:1][obj_present])

        ## class loss
        class_loss = self.ce(pred_class[obj_present], gt_class[...,0][obj_present].long())


        ## box absent
        #box_absent = self.mse(pred_box[...,1:5][obj_absent], gt_box[...,1:5][obj_absent])

        ## box loss
        #pred_box[...,1:5] = self.sig(pred_box[...,1:5])
        
        pred_box[...,1:3] = self.sig(pred_box[...,1:3])
        gt_box[...,3:5] = torch.log(1e-12 + gt_box[...,3:5])
        
        box_loss = torch.sqrt(self.mse(pred_box[...,1:5][obj_present], gt_box[...,1:5][obj_present]))
        #print( no_obj_loss.item(), obj_loss.item(), class_loss.item(), box_loss.item())
        return   self.no_obj_w * no_obj_loss + self.obj_w * obj_loss + self.class_w * class_loss + self.box_w * box_loss,  (no_obj_loss, obj_loss, class_loss, box_loss)

class LossFnCoordFocal(nn.Module):

    def __init__(self):
        super(LossFnCoordFocal, self).__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.sig = nn.Sigmoid()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()
        self.no_obj_w = 10
        self.obj_w = 1
        self.class_w = 10
        self.box_w = 10
        self.box_a = 10
        self.iou = 10

    def forward(self,  pred_class, pred_box, gt_class, gt_box):
        obj_present = gt_box[..., 0] == 1
        obj_absent = gt_box[..., 0] == 0

        # focal
        focal_loss = self.focal(pred_box[...,0:1], gt_box[...,0:1])

        ## no obj loss
        no_obj_loss = self.bce(pred_box[...,0:1][obj_absent], gt_box[...,0:1][obj_absent])

        ## obj loss
        obj_loss = self.bce(pred_box[...,0:1][obj_present], gt_box[...,0:1][obj_present])

        ## class loss
        class_loss = self.ce(pred_class[obj_present], gt_class[...,0][obj_present].long())

        
        box_loss = torch.mean(torch.square(torch.sqrt(pred_box[...,1:5][obj_present])-torch.sqrt(gt_box[...,1:5][obj_present])))
        #print(focal_loss, class_loss, box_loss)

        return focal_loss + class_loss + self.box_a * box_loss + self.no_obj_w * no_obj_loss + self.obj_w * obj_loss

