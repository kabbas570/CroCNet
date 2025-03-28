        
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch

batch_size = 1

fold = 1

path_to_checkpoints = "/data/scratch/acw676/CLIP_WORK/folds_data/weights/F" +str(fold)+ "_decoder_2gpu.pth.tar"  ## train  val

val_gt_2d_path = '/data/scratch/acw676/CLIP_WORK/folds_data/F'+str(fold)+'/LA/val/gts/'

pres_satge1_path = '/data/scratch/acw676/CLIP_WORK/folds_data/F'+str(fold)+'/F'+str(fold)+'_LA_PRED/val/pres_satge1/'
gts_path = '/data/scratch/acw676/CLIP_WORK/folds_data/F'+str(fold)+'/F'+str(fold)+'_LA_PRED/val/gts/'
imgs_path = '/data/scratch/acw676/CLIP_WORK/folds_data/F'+str(fold)+'/F'+str(fold)+'_LA_PRED/val/imgs/'

viz_gt_path = '/data/scratch/acw676/CLIP_WORK/folds_data/F'+str(fold)+'/F'+str(fold)+'_LA_PRED/val/viz_pre/'
viz_pred_path = '/data/scratch/acw676/CLIP_WORK/folds_data/F'+str(fold)+'/F'+str(fold)+'_LA_PRED/val/viz_gt/'

stage1_fs = '/data/scratch/acw676/CLIP_WORK/folds_data/F1/F1_LA_PRED/val/S1_FS_PRED/'

#val_gt_2d_path =  r'C:\My_Data\CLIP_WORK\org_data5\FOLDS_Data\F1\LA\val\temp\gts'
#path_to_checkpoints = r"C:\My_Data\ISBI_submisison\infer\F1_Base_32_Compose_withunet.pth.tar" 
#pres_satge1_path = r'C:\My_Data\ISBI_submisison\cropped_data\pres_satge1/'
#gts_path = r'C:\My_Data\ISBI_submisison\cropped_data\gts/'
#imgs_path = r'C:\My_Data\ISBI_submisison\cropped_data\imgs/'
#viz_gt_path = r'C:\My_Data\ISBI_submisison\cropped_data\viz_gt/'
#viz_pred_path = r'C:\My_Data\ISBI_submisison\cropped_data\viz_pred/'

val_loader = Data_Loader_val(val_gt_2d_path,batch_size)


### 3 - this function will save the check-points 
        #### Specify all the Hyperparameters\image dimenssions here #####



        #### Import All libraies used for training  #####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#import numpy as np
from tqdm import tqdm
import torch.optim as optim
#import matplotlib.pyplot as plt
#import pandas as pd
            ### Data_Generators ########
            
import kornia
def make_edges(image,three):
    three = np.stack((three,)*3, axis=2)
    three =torch.tensor(three)
    three = np.transpose(three, (2,0,1))  ## to bring channel first 
    three= torch.unsqueeze(three,axis = 0)
    magnitude, edges=kornia.filters.canny(three, low_threshold=0.1, high_threshold=0.2, kernel_size=(7, 7), sigma=(1, 1), hysteresis=True, eps=1e-06)
    image[np.where(edges[0,0,:,:]!=0)] = 1
    return image
    
def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def blend(image,LV,MYO,RV,three): 

    image = normalize(image)
    image = np.stack((image,)*3, axis=2)
    image[np.where(LV==1)] = [0.9,0.9,0]
    image[np.where(RV==1)] = [0,0,0.9]
    image[np.where(MYO==1)] = [0.9,0,0]
    image = make_edges(image,three)
    return image
    

from medpy import metric
def calculate_metric_percase(pred, gt):
      dice = metric.binary.dc(pred, gt)
      hd = metric.binary.hd95(pred, gt)
      return dice, hd  

from scipy.ndimage import label

def crop_largest_region_with_segmentation(image_array, segmentation_mask):
    labeled_mask, num_labels = label(segmentation_mask)
    max_region_size = 0
    largest_region = None
    for label_idx in range(1, num_labels + 1):
        rows = np.any(labeled_mask == label_idx, axis=1)
        cols = np.any(labeled_mask == label_idx, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        region_size = (ymax - ymin + 1) * (xmax - xmin + 1)
        if region_size > max_region_size:
            max_region_size = region_size
            largest_region = (ymin, ymax, xmin, xmax)
    if largest_region is not None:
        ymin, ymax, xmin, xmax = largest_region
        shift = 15
        if (ymin-shift)<0:
            shift = 1
        if (xmin-shift)<0:
            shift = 1
        cropped_region = image_array[ymin-shift:ymax+shift, xmin-shift:xmax+shift]
        return cropped_region
    else:
        return None
      
def check_Dice_Score(loader, model1, device=DEVICE):

    Dice_LV = 0
    Dice_MYO = 0
    Dice_RV = 0

    HD_LV = 0
    HD_MYO = 0
    HD_RV = 0
    
    loop = tqdm(loader)
    model1.eval()
    
    for batch_idx, (img_2d,gt_2d,name) in enumerate(loop):
    
        img_2d = img_2d.to(device=DEVICE,dtype=torch.float)  
        gt_2d = gt_2d.to(device=DEVICE,dtype=torch.float)


        
        with torch.no_grad(): 
            
            pre_2d = model1(img_2d) 
            pre_2d = (pre_2d > 0.5)*1
            
            
            #pre_2d = pre_2d.cuda().cpu()
            pre_2d = pre_2d.cpu()
            pre_2d = pre_2d.numpy().copy()
            
            #gt_2d = gt_2d.cuda().cpu()
            gt_2d = gt_2d.cpu()
            gt_2d = gt_2d.numpy().copy()
            
            img_2d = img_2d.cpu()
            img_2d = img_2d.numpy().copy()
            
            
#            fg = np.zeros(img_2d[0,0,:].shape)            
#            fg[np.where(pre_2d[0,0,:]==1)] = 1
#            fg[np.where(pre_2d[0,1,:]==1)] = 2
#            fg[np.where(pre_2d[0,2,:]==1)] = 3
            
            
            
            
            gt_blend = blend(img_2d[0,0,:],gt_2d[0,0,:],gt_2d[0,1,:],gt_2d[0,2,:],1-gt_2d[0,3,:])
            plt.imsave(viz_gt_path + name[0]+ '.png', gt_blend)
            
            pred_blend = blend(img_2d[0,0,:],pre_2d[0,0,:],pre_2d[0,1,:],pre_2d[0,2,:],1-gt_2d[0,3,:])
            plt.imsave(viz_pred_path + name[0]+ '.png', pred_blend)
            

            single_lv,single_hd_lv = calculate_metric_percase(pre_2d[0,0,:],gt_2d[0,0,:])
            single_myo,single_hd_myo = calculate_metric_percase(pre_2d[0,1,:],gt_2d[0,1,:])
            single_rv,single_hd_rv = calculate_metric_percase(pre_2d[0,2,:],gt_2d[0,2,:])
            
            
            cropped_img = crop_largest_region_with_segmentation(img_2d[0,0,:],1-pre_2d[0,3,:])
            print(cropped_img.shape)
            cropped_img = sitk.GetImageFromArray(cropped_img) 
            sitk.WriteImage(cropped_img,imgs_path+name[0]+'.nii.gz')
            
            cropped_gt = np.zeros(img_2d[0,0,:].shape)            
            cropped_gt[np.where(gt_2d[0,0,:]==1)] = 1
            cropped_gt[np.where(gt_2d[0,1,:]==1)] = 2
            cropped_gt[np.where(gt_2d[0,2,:]==1)] = 3
            cropped_gt = crop_largest_region_with_segmentation(cropped_gt,1-pre_2d[0,3,:])  
            cropped_gt = sitk.GetImageFromArray(cropped_gt)
            sitk.WriteImage(cropped_gt,gts_path+name[0]+'.nii.gz')
            
            
            
            cropped_pre = np.zeros(img_2d[0,0,:].shape)            
            cropped_pre[np.where(pre_2d[0,0,:]==1)] = 1
            cropped_pre[np.where(pre_2d[0,1,:]==1)] = 2
            cropped_pre[np.where(pre_2d[0,2,:]==1)] = 3 
            
            plt.imsave(stage1_fs + name[0]+ '.png', cropped_pre)
                       
            cropped_pre = crop_largest_region_with_segmentation(cropped_pre,1-pre_2d[0,3,:])
            cropped_pre = sitk.GetImageFromArray(cropped_pre)
            sitk.WriteImage(cropped_pre,pres_satge1_path+name[0]+'.nii.gz') 
            
            
            Dice_LV+=single_lv
            HD_LV+=single_hd_lv#
            
            Dice_MYO+=single_myo
            HD_MYO+=single_hd_myo
            
            
            Dice_RV+=single_rv
            HD_RV+=single_hd_rv
            
        
    print('For LV','      ',Dice_LV/len(loader),'    ',HD_LV/len(loader))
    print('For MYO','      ',Dice_MYO/len(loader),'    ',HD_MYO/len(loader))
    print('For RV','      ',Dice_RV/len(loader),'    ',HD_RV/len(loader))


model_ = UNet_att()

def eval_():
    model = model_.to(device=DEVICE,dtype=torch.float)
    # optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=0)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9),lr=0)
    
    checkpoint = torch.load(path_to_checkpoints,map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    print('Dice Score for LA')
    _ = check_Dice_Score(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    eval_()
