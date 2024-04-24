import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  #### Specify all the paths here #####
   

path_to_save_Learning_Curve = '/data/scratch/acw676/CLIP_WORK/folds_data/weights/'+'/config_22_F5'
path_to_save_check_points = '/data/scratch/acw676/CLIP_WORK/folds_data/weights/'+'/config_22_F5'
### 3 - this function will save the check-points 
def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
        #### Specify all the Hyperparameters\image dimenssions here #####

batch_size = 8
Max_Epochs = 150

        #### Import All libraies used for training  #####



val_gt_2d_path = '/data/scratch/acw676/CLIP_WORK/folds_data/F5/F5_LA_PRED/val/gts/'
train_gt_2d_path = '/data/scratch/acw676/CLIP_WORK/folds_data/F5/F5_LA_PRED/train/gts/'

val_loader = Data_Loader_val(val_gt_2d_path,batch_size)
train_loader = Data_Loader_train(train_gt_2d_path,batch_size)

# a = iter(train_loader)

# for i in range(320):
#     a1 = next(a)



   ### Load the Data using Data generators and paths specified #####
   #######################################
   
print(len(train_loader)) ### this shoud be = Total_images/ batch size
print(len(val_loader))   ### same here
#print(len(test_loader))   ### same here

### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve

avg_train_losses1_seg = []   # losses of all training epochs
avg_valid_losses1_seg = []  #losses of all training epochs

avg_valid_DS_ValSet = []  # all training epochs
avg_valid_DS_TrainSet = []  # all training epochs
### Next we have all the funcitons which will be called in the main for training ####



def check_Dice_Score(loader, model1, device=DEVICE):
    
    Dice_score_LA_RV = 0
    Dice_score_LA_MYO = 0
    Dice_score_LA_LV = 0
    
    loop = tqdm(loader)
    model1.eval()
    
    for batch_idx, (img_lv,img_myo,img_rv,gt_2d,name) in enumerate(loop):
        
        img_lv = img_lv.to(device=DEVICE,dtype=torch.float)  
        img_myo = img_myo.to(device=DEVICE,dtype=torch.float)  
        img_rv = img_rv.to(device=DEVICE,dtype=torch.float)  
        gt_2d = gt_2d.to(device=DEVICE,dtype=torch.float)
        
        with torch.no_grad(): 
            pre_lv,pre_myo,pre_rv = model1(img_lv,img_myo,img_rv) 
            
            pre_lv = (pre_lv > 0.5)*1
            pre_myo = (pre_myo > 0.5)*1
            pre_rv = (pre_rv > 0.5)*1

            out_LV = pre_lv
            out_MYO = pre_myo
            out_RV = pre_rv
            
            ## Dice Score for ES-LA ###

            single_LA_LV = (2 * (out_LV * gt_2d[:,0:1,:]).sum()) / (
               (out_LV + gt_2d[:,0:1,:]).sum() + 1e-8)
           
            Dice_score_LA_LV +=single_LA_LV
           
            single_LA_MYO = (2 * (out_MYO * gt_2d[:,1:2,:]).sum()) / (
   (out_MYO + gt_2d[:,1:2,:]).sum() + 1e-8)
            
            Dice_score_LA_MYO += single_LA_MYO

            single_LA_RV = (2 * (out_RV * gt_2d[:,2:3,:]).sum()) / (
       (out_RV + gt_2d[:,2:3,:]).sum() + 1e-8)
            
            Dice_score_LA_RV += single_LA_RV
            

    
    print(f"Dice_score_LA_RV  : {Dice_score_LA_RV/len(loader)}")
    print(f"Dice_score_LA_MYO  : {Dice_score_LA_MYO/len(loader)}")
    print(f"Dice_score_LA_LV  : {Dice_score_LA_LV/len(loader)}")

    Overall_Dicescore_LA =( Dice_score_LA_RV + Dice_score_LA_MYO + Dice_score_LA_LV ) /3
    
    return Overall_Dicescore_LA/len(loader)


### 2- the main training fucntion to update the weights....
def train_fn(loader_train1,loader_valid1,model1, optimizer1, scaler):  ### Loader_1--> ED and Loader2-->ES
    train_losses1_seg  = [] # loss of each batch
    valid_losses1_seg  = []  # loss of each batch

    loop = tqdm(loader_train1)
    model1.train()
    
    for batch_idx, (img_lv,img_myo,img_rv,gt_2d,name) in enumerate(loop):
        
        img_lv = img_lv.to(device=DEVICE,dtype=torch.float)  
        img_myo = img_myo.to(device=DEVICE,dtype=torch.float)  
        img_rv = img_rv.to(device=DEVICE,dtype=torch.float)  
        gt_2d = gt_2d.to(device=DEVICE,dtype=torch.float)

        with torch.cuda.amp.autocast():
            pre_lv,pre_myo,pre_rv = model1(img_lv,img_myo,img_rv) 
            loss1 = loss_fn_DC(pre_lv,gt_2d[:,0:1,:])
            loss2 = loss_fn_DC(pre_myo,gt_2d[:,1:2,:])
            loss3 = loss_fn_DC(pre_rv,gt_2d[:,2:3,:])

        # backward
        
        loss = (loss1 + loss2 + loss3)/3
        
        optimizer1.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer1)

        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        train_losses1_seg.append(float(loss))
        
    loop_v = tqdm(loader_valid1)
    model1.eval() 
    
    for batch_idx, (img_lv,img_myo,img_rv,gt_2d,name) in enumerate(loop_v):
        
        img_lv = img_lv.to(device=DEVICE,dtype=torch.float)  
        img_myo = img_myo.to(device=DEVICE,dtype=torch.float)  
        img_rv = img_rv.to(device=DEVICE,dtype=torch.float)  
        gt_2d = gt_2d.to(device=DEVICE,dtype=torch.float)

        with torch.no_grad(): 
            pre_lv,pre_myo,pre_rv = model1(img_lv,img_myo,img_rv) 
            loss1 = loss_fn_DC(pre_lv,gt_2d[:,0:1,:])
            loss2 = loss_fn_DC(pre_myo,gt_2d[:,1:2,:])
            loss3 = loss_fn_DC(pre_rv,gt_2d[:,2:3,:]) 

        # backward
        loss = (loss1 + loss2 + loss3)/3
        loop_v.set_postfix(loss = loss.item())
        valid_losses1_seg.append(float(loss))

    train_loss_per_epoch1_seg = np.average(train_losses1_seg)
    valid_loss_per_epoch1_seg  = np.average(valid_losses1_seg)
    
    
    avg_train_losses1_seg.append(train_loss_per_epoch1_seg)
    avg_valid_losses1_seg.append(valid_loss_per_epoch1_seg)
    
    return train_loss_per_epoch1_seg , valid_loss_per_epoch1_seg


model_1 = My_Net1()


epoch_len = len(str(Max_Epochs))

                          ## loss fucniton    ####
import torch
import torch.nn as nn

   
# W_1 = torch.tensor([1,1,1,1])  
# W_1 = W_1.to(device=DEVICE,dtype=torch.float)

W_1 =None

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=W_1, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self,weight=W_1, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

    
def compute_per_channel_dice(input, target, epsilon=1e-6, weight=W_1):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))   
    
    
    ## loss fucniton    ####
    
    
    
loss_fn_DC = DiceLoss()


def main():
    model1 = model_1.to(device=DEVICE,dtype=torch.float)
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Max_Epochs):
        
        if epoch<=10:
          LEARNING_RATE = 0.0001
          
        if epoch>10:
          LEARNING_RATE = 0.00005
        
        if epoch>50:
          LEARNING_RATE = 0.00005
          
        if epoch>80:
          LEARNING_RATE = 0.00008
          
        if epoch>100:
          LEARNING_RATE = 0.00001
        
        # if epoch>200:
        #   LEARNING_RATE = 0.000005
        
        # if epoch>300:
        #   LEARNING_RATE = 0.000001
          
        #optimizer1 = optim.SGD(model1.parameters(),lr=LEARNING_RATE)
        optimizer1 = optim.Adam(model1.parameters(),betas=(0.9, 0.99),lr=LEARNING_RATE)
        train_loss,valid_loss = train_fn(train_loader,val_loader, model1, optimizer1,scaler)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        Dice_val = check_Dice_Score(val_loader, model1, device=DEVICE)
        avg_valid_DS_ValSet.append(Dice_val.detach().cpu().numpy())
        
            # save model
    checkpoint = {
        "state_dict": model1.state_dict(),
        "optimizer":optimizer1.state_dict(),
    }
    save_checkpoint(checkpoint)
        

if __name__ == "__main__":
    main()

### This part of the code will generate the learning curve ......

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
#plt.plot(range(1,len(avg_train_losses1_clip)+1),avg_train_losses1_clip, label='Training CLIP Loss')
#plt.plot(range(1,len(avg_valid_losses1_clip)+1),avg_valid_losses1_clip,label='Validation CLIP Loss')

plt.plot(range(1,len(avg_train_losses1_seg)+1),avg_train_losses1_seg, label='Training Segmentation Loss')
plt.plot(range(1,len(avg_valid_losses1_seg)+1),avg_valid_losses1_seg,label='Validation Segmentation Loss')

plt.plot(range(1,len(avg_valid_DS_ValSet)+1),avg_valid_DS_ValSet,label='Validation DS')
plt.plot(range(1,len(avg_valid_DS_TrainSet)+1),avg_valid_DS_TrainSet,label='Train DS')

# find position of lowest validation loss
minposs = avg_valid_losses1_seg.index(min(avg_valid_losses1_seg))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')
font1 = {'size':20}
plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(avg_train_losses1_seg)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')
