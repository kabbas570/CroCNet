
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import os
import copy
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import to_2tuple
import einops
import torch.nn.functional as F


class Mlp_2d(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class X_EfficientAdditiveAttnetion(nn.Module):

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

            ###  For LV ###
            
        self.to_query_lv = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key_lv = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g_lv = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor_lv = token_dim ** -0.5
        self.Proj_lv1 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.Proj_lv2 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.Proj_lv3 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final_lv = nn.Linear(token_dim * num_heads, token_dim)
        
        ###  For MYO ###
        
        self.to_query_myo = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key_myo = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g_myo= nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor_myo = token_dim ** -0.5
        self.Proj_myo1 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.Proj_myo2 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.Proj_myo3 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        
        self.final_myo = nn.Linear(token_dim * num_heads, token_dim)
        
        
        ###  For RV ###
        
        self.to_query_rv = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key_rv = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g_rv = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor_rv = token_dim ** -0.5
        self.Proj_rv1 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.Proj_rv2 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.Proj_rv3 = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        
        self.final_rv = nn.Linear(token_dim * num_heads, token_dim)
        
        

    def forward(self, x_lv,x_myo,x_rv):
        
        # ###  For 2D ###
        
        query_lv = self.to_query_lv(x_lv)
        key_lv = self.to_key_lv(x_lv)
        
        query_myo = self.to_query_myo(x_myo)
        key_myo = self.to_key_myo(x_myo)
        
        query_rv = self.to_query_rv(x_rv)
        key_rv = self.to_key_rv(x_rv)
        
        
        query_lv = torch.nn.functional.normalize(query_lv, dim=-1) #BxNxD
        key_lv = torch.nn.functional.normalize(key_lv, dim=-1) #BxNxD
        
        query_myo = torch.nn.functional.normalize(query_myo, dim=-1) #BxNxD
        key_myo = torch.nn.functional.normalize(key_myo, dim=-1) #BxNxD
        
        query_rv = torch.nn.functional.normalize(query_rv, dim=-1) #BxNxD
        key_rv = torch.nn.functional.normalize(key_rv, dim=-1) #BxNxD
        
        
        query_weight_lv = query_lv @ self.w_g_lv # BxNx1 (BxNxD @ Dx1)
        A_lv = query_weight_lv * self.scale_factor_lv # BxNx1

        A_lv = torch.nn.functional.normalize(A_lv, dim=1) # BxNx1

        G_lv = torch.sum(A_lv * query_lv, dim=1) # BxD

        G_lv = einops.repeat(
            G_lv, "b d -> b repeat d", repeat=key_lv.shape[1]
        ) # BxNxD
        
        out_lv= self.Proj_lv1(G_lv * key_lv) + self.Proj_lv2(G_lv * key_myo) + self.Proj_lv3(G_lv * key_rv)  + query_lv #BxNxD
        
        out_lv = self.final_lv(out_lv) # BxNxD
        
        ## MYO
        query_weight_myo = query_myo @ self.w_g_myo # BxNx1 (BxNxD @ Dx1)
        A_myo = query_weight_myo * self.scale_factor_myo # BxNx1

        A_myo = torch.nn.functional.normalize(A_myo, dim=1) # BxNx1

        G_myo = torch.sum(A_myo * query_myo, dim=1) # BxD

        G_myo= einops.repeat(
            G_myo, "b d -> b repeat d", repeat=key_myo.shape[1]
        ) # BxNxD
        
        out_myo = self.Proj_myo1(G_myo * key_lv) + self.Proj_myo2(G_myo * key_myo) + self.Proj_myo3(G_myo * key_rv)+ query_myo #BxNxD
        
        out_myo = self.final_myo(out_myo) # BxNxD
        
        ## MYO
        query_weight_rv = query_rv @ self.w_g_rv # BxNx1 (BxNxD @ Dx1)
        A_rv = query_weight_rv * self.scale_factor_rv # BxNx1

        A_rv = torch.nn.functional.normalize(A_rv, dim=1) # BxNx1

        G_rv = torch.sum(A_rv * query_rv, dim=1) # BxD

        G_rv = einops.repeat(
            G_rv, "b d -> b repeat d", repeat=key_rv.shape[1]
        ) # BxNxD
        
        out_rv = self.Proj_rv1(G_rv * key_lv) + self.Proj_rv2(G_rv * key_myo) + self.Proj_rv3(G_rv * key_rv) + query_rv #BxNxD
        
        out_rv = self.final_myo(out_rv) # BxNxD
        

        return out_lv,out_myo,out_rv
        


class My_NetXd(nn.Module):
    def __init__(self, out=16):
        super(My_NetXd, self).__init__()
        
        self.attn_X = X_EfficientAdditiveAttnetion(in_dims=out, token_dim=out, num_heads=1)
        
        
        drop_path=0.
        layer_scale_init_value=1e-5
        
        self.drop_path_lv = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
            
        self.drop_path_myo = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
            
        
        self.drop_path_rv = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
            
        self.layer_scale_1_lv = nn.Parameter(
            layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        
        self.layer_scale_2_lv = nn.Parameter(
            layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        
        
        self.layer_scale_1_myo = nn.Parameter(
            layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        
        self.layer_scale_2_myo = nn.Parameter(
            layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        
        
        self.layer_scale_1_rv = nn.Parameter(
            layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        
        self.layer_scale_2_rv = nn.Parameter(
            layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        
        
        self.linear_2d_lv = Mlp_2d(in_features=out, hidden_features=int(out * 4.0), act_layer=nn.GELU, drop=0.)
        self.linear_2d_myo = Mlp_2d(in_features=out, hidden_features=int(out * 4.0), act_layer=nn.GELU, drop=0.)
        self.linear_2d_rv = Mlp_2d(in_features=out, hidden_features=int(out * 4.0), act_layer=nn.GELU, drop=0.)
                
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        
    def forward(self,x_lv,x_myo,x_rv):
        
        
        B1,C1, H1, W1 = x_lv.shape
        
        x1_lv,x1_myo,x1_rv = self.attn_X(x_lv.permute(0, 2, 3, 1).reshape(B1, H1 * W1, C1),x_myo.permute(0, 2, 3, 1).reshape(B1, H1 * W1, C1),x_rv.permute(0, 2, 3, 1).reshape(B1, H1 * W1, C1))

        x_lv = x_lv + self.drop_path_lv(
        self.layer_scale_1_lv * x1_lv.reshape(B1, H1, W1, C1).permute(
            0, 3, 1, 2))
        x_lv = x_lv + self.drop_path_lv(self.layer_scale_2_lv * self.linear_2d_lv(x_lv))
        
        
        x_myo = x_myo + self.drop_path_myo(
        self.layer_scale_1_myo * x1_myo.reshape(B1, H1, W1, C1).permute(
            0, 3, 1, 2))
        x_myo = x_myo + self.drop_path_myo(self.layer_scale_2_myo * self.linear_2d_myo(x_myo))
        
        
        x_rv = x_rv + self.drop_path_rv(
        self.layer_scale_1_rv * x1_rv.reshape(B1, H1, W1, C1).permute(
            0, 3, 1, 2))
        x_rv = x_rv + self.drop_path_rv(self.layer_scale_2_rv * self.linear_2d_rv(x_rv))
        
        
        return  x_lv,x_myo,x_rv
        

# Input_Image_Channels = 16
# def model() -> My_NetXd:
#     model = My_NetXd()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256),(Input_Image_Channels, 256,256),(Input_Image_Channels, 256,256)])   
        
        
def stem_2d(in_chs, out_chs):

    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=1, padding=1), ### using stride 2 Twice 
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )

class SwiftFormerLocalRepresentation_2d(nn.Module):

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x
    
 
class EfficientAdditiveAttnetion(nn.Module):

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)
        
    def forward(self, x):
        
        #print(x.shape)
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD
        
        out = self.Proj(G * key) + query #BxNxD

        out = self.final(out) # BxNxD

        return out




class Embedding_2d(nn.Module):

    def __init__(self, patch_size=3, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(1)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=2, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
      


class ConvEncoder_2d(nn.Module):

    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x
    

    
    
class My_Net2d(nn.Module):
    def __init__(self, n_channels = 1,out=32):
        super(My_Net2d, self).__init__()
        
        
        drop_path=0.
        layer_scale_init_value=1e-5
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
            
        
        self.local_representation1_2d = SwiftFormerLocalRepresentation_2d(dim=out, kernel_size=3, drop_path=0.,
                                                                   use_layer_scale=True)
        
        self.attn_2d = EfficientAdditiveAttnetion(in_dims=out, token_dim=out, num_heads=1)
        self.linear_2d = Mlp_2d(in_features=out, hidden_features=int(out * 4.0), act_layer=nn.GELU, drop=0.)
        self.layer_scale_1_2d = nn.Parameter(
            layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.layer_scale_2_2d = nn.Parameter(
            layer_scale_init_value * torch.ones(out).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.embd_2d_1 = Embedding_2d(in_chans=out, embed_dim=out*2)
        
        
        self.encoder1 = ConvEncoder_2d(dim=out, hidden_dim=int(4.0 * out), kernel_size=3)
        
                        
    def forward(self,x_2d):
        
        
        x_2d = self.encoder1(x_2d)
        
        x1_2d = self.local_representation1_2d(x_2d)
        B, C, H, W = x1_2d.shape
        x1_2d = x1_2d + self.drop_path(
            self.layer_scale_1_2d * self.attn_2d(x1_2d.permute(0, 2, 3, 1).reshape(B, H * W, C)).reshape(B, H, W, C).permute(
                0, 3, 1, 2))
        x1_2d = x1_2d + self.drop_path(self.layer_scale_2_2d * self.linear_2d(x1_2d))
        x1_2d = self.embd_2d_1(x1_2d)
        
        return x1_2d


        
class DoubleConv_2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Up_2d(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


Base = 32

class My_Net1(nn.Module):
    def __init__(self, n_channels = 2, bilinear=False):
        super(My_Net1, self).__init__()
        
        factor = 1
                
        self.block2d_1 = My_Net2d(n_channels=1,out=Base)
        self.block2d_2 = My_Net2d(n_channels=2*Base,out=2*Base)
        self.block2d_3 = My_Net2d(n_channels=4*Base,out=4*Base)
        self.block2d_4 = My_Net2d(n_channels=8*Base,out=8*Base)
        #self.block2d_5 = My_Net2d(n_channels=16*Base,out=16*Base)
        
        self.up1_2d = Up_2d(32*Base, 16*Base // factor)
        self.up2_2d = Up_2d(16*Base, 8*Base // factor)
        self.up3_2d = Up_2d(8*Base, 4*Base // factor)
        self.up4_2d = Up_2d(4*Base, 2*Base, bilinear)
        self.up5_2d = Up_2d(2*Base, Base, bilinear)
        self.outc_2d_lv = OutConv_2d(Base, 1)
        #self.outc_2d_rv = OutConv_2d(Base, 1)
        #self.outc_2d_myo = OutConv_2d(Base, 1)
       
        self.stem_2d = stem_2d(n_channels,Base) 
        
        self.X_0 = My_NetXd(out=Base)
        self.X_1 = My_NetXd(out=2*Base)
        self.X_2 = My_NetXd(out=4*Base)
        self.X_3 = My_NetXd(out=8*Base)
        self.X_4 = My_NetXd(out=16*Base)
        
        #self.drop = nn.Dropout(p=0.10)
        
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        
    def forward(self, x_lv,x_myo,x_rv):
        
        
        ## Encoders ####

        x0_lv = self.stem_2d(x_lv)
        x0_myo = self.stem_2d(x_myo)
        x0_rv = self.stem_2d(x_rv)
        
        x0_lv,x0_myo,x0_rv = self.X_0(x0_lv,x0_myo,x0_rv)
        
        x1_lv = self.block2d_1(x0_lv)
        x1_myo = self.block2d_1(x0_myo)
        x1_rv = self.block2d_1(x0_rv)
        x1_lv,x1_myo,x1_rv = self.X_1(x1_lv,x1_myo,x1_rv)
        
        
        x2_lv = self.block2d_2(x1_lv)
        x2_myo = self.block2d_2(x1_myo)
        x2_rv = self.block2d_2(x1_rv)
        x2_lv,x2_myo,x2_rv = self.X_2(x2_lv,x2_myo,x2_rv)
        
        
        
        x3_lv = self.block2d_3(x2_lv)
        x3_myo = self.block2d_3(x2_myo)
        x3_rv = self.block2d_3(x2_rv)
        x3_lv,x3_myo,x3_rv = self.X_3(x3_lv,x3_myo,x3_rv)
        
        
        x4_lv = self.block2d_4(x3_lv)
        x4_myo = self.block2d_4(x3_myo)
        x4_rv = self.block2d_4(x3_rv)
        x4_lv,x4_myo,x4_rv = self.X_4(x4_lv,x4_myo,x4_rv)
        
#        x4_lv = self.drop(x4_lv)
#        x4_myo = self.drop(x4_myo)
#        x4_rv = self.drop(x4_rv)

        ## Decoders ####
        
        x2lv = self.up2_2d(x4_lv, x3_lv) 
        x2lv = self.up3_2d(x2lv, x2_lv) 
        x2lv = self.up4_2d(x2lv, x1_lv) 
        x2lv = self.up5_2d(x2lv, x0_lv) 
        

        x2myo = self.up2_2d(x4_myo, x3_myo) 
        x2myo = self.up3_2d(x2myo, x2_myo) 
        x2myo = self.up4_2d(x2myo, x1_myo) 
        x2myo = self.up5_2d(x2myo, x0_myo) 

        x2rv = self.up2_2d(x4_rv, x3_rv) 
        x2rv = self.up3_2d(x2rv, x2_rv) 
        x2rv = self.up4_2d(x2rv, x1_rv)
        x2rv = self.up5_2d(x2rv, x0_rv) 
        
        return self.outc_2d_lv(x2lv), self.outc_2d_lv(x2myo) , self.outc_2d_lv(x2rv)

# Input_Image_Channels = 2
# def model() -> My_Net1:
#     model = My_Net1()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 128,128),(Input_Image_Channels, 128,128),(Input_Image_Channels, 128,128)])
