from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import torchio as tio
import torch

NUM_WORKERS=0
PIN_MEMORY=True

DIM_ = 128

def same_depth(img):
    temp = np.zeros([img.shape[0],36,DIM_,DIM_])
    temp[:,0:img.shape[1],:,:] = img
    return temp 

def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(img_):
    
    org_dim3 = img_.shape[0]
    org_dim1 = img_.shape[1]
    org_dim2 = img_.shape[2] 
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<=DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_

def normalize_Mstd(image):
        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        return image


transforms_2d = tio.OneOf({
        tio.RandomFlip(axes=([1,2])): .2,  ## axis [0,1] or [1,2]
        tio.RandomElasticDeformation(num_control_points=(5,5,5),locked_borders=1,image_interpolation ='nearest'): 0.2,
        tio.RandomAffine(degrees=(-30,30),center='image'): 0.2, ## for 2D rotation
        
        tio.RandomBlur(): 0.2,
        tio.RandomGamma(log_gamma=(-0.2, -0.2)): 0.2, 
        tio.RandomNoise(mean=0.1,std=0.1):0.2,
})

#transforms_2d = tio.Compose({
#        tio.RandomFlip(axes=([1,2])): .1,  ## axis [0,1] or [1,2]
#        tio.RandomElasticDeformation(num_control_points=(5,5,5),locked_borders=1,image_interpolation ='nearest'): 0.1,
#        tio.RandomAffine(degrees=(-45,45),center='image'): 0.1, ## for 2D rotation
#        
#        tio.RandomBlur(): 0.1,
#        tio.RandomGamma(log_gamma=(-0.2, -0.2)): 0.1, 
#        tio.RandomNoise(mean=0.1,std=0.1):0.1,
#})
#           

class Dataset_train(Dataset):
    def __init__(self, data_folder_2d,transformations_2d=transforms_2d):
        self.data_folder_2d = data_folder_2d
        self.images = os.listdir(data_folder_2d)
        self.transformations_2d = transformations_2d
       
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        gt_2d_path = os.path.join(self.data_folder_2d, self.images[index])
        gt_2d = sitk.ReadImage(gt_2d_path)
        gt_2d = sitk.GetArrayFromImage(gt_2d)
        gt_2d = np.expand_dims(gt_2d, axis=0)
        gt_2d = Cropping_3d(gt_2d)
        gt_2d = np.expand_dims(gt_2d, axis=3)  ## adding the depth dim for torch.io

        
        img_2d_path = gt_2d_path.replace("gt", "img") 
        img_2d = sitk.ReadImage(img_2d_path)
        img_2d = sitk.GetArrayFromImage(img_2d)
        img_2d = np.expand_dims(img_2d, axis=0)
        img_2d = Cropping_3d(img_2d)
        img_2d = np.expand_dims(img_2d, axis=3)  ## adding the depth dim for torch.io
        
        
        pre_2d_path = img_2d_path.replace("imgs", "pres_satge1") 
        pre_2d = sitk.ReadImage(pre_2d_path)
        pre_2d = sitk.GetArrayFromImage(pre_2d)
        pre_2d = np.expand_dims(pre_2d, axis=0)
        pre_2d = Cropping_3d(pre_2d)
        pre_2d = np.expand_dims(pre_2d, axis=3)  ## adding the depth dim for torch.io
        
        
        
        d = {}
        d['Image'] = tio.Image(tensor = img_2d, type=tio.INTENSITY)
        d['Mask1'] = tio.Image(tensor = gt_2d, type=tio.LABEL)
        d['Mask2'] = tio.Image(tensor = pre_2d, type=tio.LABEL)
        sample = tio.Subject(d)
        if self.transformations_2d is not None:
            transformed_tensor = self.transformations_2d(sample)
            img_2d = transformed_tensor['Image'].data
            gt_2d = transformed_tensor['Mask1'].data
            pre_2d = transformed_tensor['Mask2'].data
            
    
        gt_2d = gt_2d[:,:,:,0]
        pre_2d = pre_2d[:,:,:,0]
        img_2d = img_2d[:,:,:,0]

        
        temp_2d = np.zeros([4,DIM_,DIM_])
        temp_2d[0:1,:][np.where(gt_2d==1)]=1
        temp_2d[1:2,:][np.where(gt_2d==2)]=1
        temp_2d[2:3,:][np.where(gt_2d==3)]=1
        temp_2d[3:4,:][np.where(gt_2d==0)]=1
        
        
        temp_2d_pred = np.zeros([4,DIM_,DIM_])
        temp_2d_pred[0:1,:][np.where(pre_2d==1)]=1
        temp_2d_pred[1:2,:][np.where(pre_2d==2)]=1
        temp_2d_pred[2:3,:][np.where(pre_2d==3)]=1
        temp_2d_pred[3:4,:][np.where(pre_2d==0)]=1
        
        
        img_lv = np.concatenate((img_2d, temp_2d_pred[0:1,:]), 0)
        img_myo = np.concatenate((img_2d, temp_2d_pred[1:2,:]), 0)
        img_rv = np.concatenate((img_2d, temp_2d_pred[2:3,:]), 0)


        
        return img_lv,img_myo,img_rv,temp_2d,self.images[index][:-4]
    
def Data_Loader_train(data_folder_2d,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_train( data_folder_2d=data_folder_2d)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


class Dataset_val(Dataset):
    def __init__(self, data_folder_2d):
        self.data_folder_2d = data_folder_2d
        self.images = os.listdir(data_folder_2d)
       
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        gt_2d_path = os.path.join(self.data_folder_2d, self.images[index])
        gt_2d = sitk.ReadImage(gt_2d_path)
        gt_2d = sitk.GetArrayFromImage(gt_2d)
        gt_2d = np.expand_dims(gt_2d, axis=0)
        gt_2d = Cropping_3d(gt_2d)

        
        img_2d_path = gt_2d_path.replace("gt", "img") 
        img_2d = sitk.ReadImage(img_2d_path)
        img_2d = sitk.GetArrayFromImage(img_2d)
        img_2d = np.expand_dims(img_2d, axis=0)
        img_2d = Cropping_3d(img_2d)
        
        
        pre_2d_path = img_2d_path.replace("imgs", "pres_satge1") 
        pre_2d = sitk.ReadImage(pre_2d_path)
        pre_2d = sitk.GetArrayFromImage(pre_2d)
        pre_2d = np.expand_dims(pre_2d, axis=0)
        pre_2d = Cropping_3d(pre_2d)
        
        temp_2d = np.zeros([4,DIM_,DIM_])
        temp_2d[0:1,:][np.where(gt_2d==1)]=1
        temp_2d[1:2,:][np.where(gt_2d==2)]=1
        temp_2d[2:3,:][np.where(gt_2d==3)]=1
        temp_2d[3:4,:][np.where(gt_2d==0)]=1
        
        
        temp_2d_pred = np.zeros([4,DIM_,DIM_])
        temp_2d_pred[0:1,:][np.where(pre_2d==1)]=1
        temp_2d_pred[1:2,:][np.where(pre_2d==2)]=1
        temp_2d_pred[2:3,:][np.where(pre_2d==3)]=1
        temp_2d_pred[3:4,:][np.where(pre_2d==0)]=1

        
        img_lv = np.concatenate((img_2d, temp_2d_pred[0:1,:]), 0)
        img_myo = np.concatenate((img_2d, temp_2d_pred[1:2,:]), 0)
        img_rv = np.concatenate((img_2d, temp_2d_pred[2:3,:]), 0)

        return img_lv,img_myo,img_rv,temp_2d,self.images[index][:-4]
    
    
def Data_Loader_val(data_folder_2d,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val( data_folder_2d=data_folder_2d)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader
