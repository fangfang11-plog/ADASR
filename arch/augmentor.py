import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import args



def get_rot_mat(theta):
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    row1 = torch.cat((cos,-sin),dim=0)
    row1 = torch.cat((row1,torch.zeros(1).to(args.device)),dim=0).unsqueeze(0)
    row2 = torch.cat((sin,cos),dim=0)
    row2 = torch.cat((row2,torch.zeros(1).to(args.device)),dim=0).unsqueeze(0)
    result = torch.cat((row1,row2),dim=0)

    return result

def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)

    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)

    return x

class Augmentor_Rotation(nn.Module):
    def __init__(self):
        super(Augmentor_Rotation, self).__init__()
        #标注
        self.fc1 = nn.Linear(48*48,512)
        # self.bn1 = nn.BatchNorm1d(24*24)
        # self.fc2 = nn.Linear(24*24,128)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64,16)
        # self.fc5 = nn.Linear(16,4)
        self.fc6 = nn.Linear(512,1)

    def forward(self,input):
        x = torch.relu(self.fc1(input))
        # x = F.relu(self.fc5(x))
        # x = torch.sigmoid(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))
        # x = torch.sigmoid(self.fc4(x))
        # x = torch.sigmoid(self.fc5(x))
        y = self.fc6(x)
        return y


class Augmentor1(nn.Module):
    def __init__(self,dim,in_dim=3):
        super(Augmentor1, self).__init__()
        #self.conv1 = nn.Conv2d(dim,64,1)
        self.conv2 = nn.Conv2d(dim,128,1)
        self.conv3 = nn.Conv2d(128,256,1)
        self.conv4 = nn.Conv2d(256,512,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.rot = Augmentor_Rotation()
    def forward(self,img):
        B,C,H,W = img.size()

        x = img
        x = F.sigmoid(self.bn2(self.conv2(x)))

        x = F.sigmoid(self.bn3(self.conv3(x)))

        x = F.sigmoid(self.bn4(self.conv4(x)))
        
        x = torch.max(x,1,keepdim=True)[0]

        feat_r = x.view(-1,)
        angle = self.rot(feat_r)

        aug_img = rot_img(img,angle * 2 * np.pi,dtype=torch.cuda.FloatTensor)
        return aug_img,angle * 2 * np.pi


# class Augmentor(nn.Module):
#     def __init__(self,dim):
#         super(Augmentor, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(dim, 64, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 24
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), #12
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), #6
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(512, 512, kernel_size=6, stride=2, padding=1), # 3
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(512, 1024, kernel_size=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(1024, 1, kernel_size=1)
#         )
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         angle = F.sigmoid(self.net(x).view(batch_size))
#
#         aug_img = rot_img(x,angle * 2 * np.pi,dtype=torch.cuda.FloatTensor)
#
#         return aug_img,angle * 2 * np.pi


import torch
import torch.nn as nn
import torch.nn.functional as F

class Augmentor(nn.Module):
    def __init__(self, in_channels):
        super(Augmentor, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=64)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=64)
        self.relu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(num_features=128)
        self.relu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.bn4 = nn.BatchNorm3d(num_features=128)
        self.relu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), padding=1)
        self.bn5 = nn.BatchNorm3d(num_features=256)
        self.relu5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.bn6 = nn.BatchNorm3d(num_features=256)
        self.relu6 = nn.LeakyReLU(0.2)

        self.conv7 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), padding=1)
        self.bn7 = nn.BatchNorm3d(num_features=512)
        self.relu7 = nn.LeakyReLU(0.2)

        self.conv8 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.bn8 = nn.BatchNorm3d(num_features=512)
        self.relu8 = nn.LeakyReLU(0.2)

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.relu9 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(in_features=1024, out_features=1)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)

        x = self.avgpool(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.relu9(x)
        x = self.fc2(x)

        angle = F.tanh(x).squeeze(0)

        return angle * 3.14# * 2