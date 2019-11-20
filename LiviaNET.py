from Blocks import *
import torch.nn.init as init
import torch.nn.functional as F
import pdb
import math
#from layers import *

def croppCenter(tensorToCrop,finalShape):

    org_shape = tensorToCrop.shape
    diff = org_shape[2] - finalShape[2]
    croppBorders = int(diff/2)
    return tensorToCrop[:,
                        :,
                        croppBorders:org_shape[2]-croppBorders,
                        croppBorders:org_shape[3]-croppBorders,
                        croppBorders:org_shape[4]-croppBorders]

def convBlock(nin, nout, kernel_size=3, batchNorm = False, layer=nn.Conv3d, bias=True, dropout_rate = 0.0, dilation = 1):
    
    if batchNorm == False:
        return nn.Sequential(
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, dilation=dilation)
        )
    else:
        return nn.Sequential(
            nn.BatchNorm3d(nin),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, dilation=dilation)
        )
        
def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation = 1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        #nn.LeakyReLU(0.2)
        nn.PReLU()
    )

class LiviaNet(nn.Module):
    def __init__(self, nClasses):
        super(LiviaNet, self).__init__()
        
        # Path-Top
        #self.conv1_Top = torch.nn.Conv3d(1, 25, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1_Top = convBlock(1, 25)
        self.conv2_Top = convBlock(25, 25, batchNorm = True)
        self.conv3_Top = convBlock(25, 25, batchNorm = True)
        self.conv4_Top = convBlock(25, 50, batchNorm = True)
        self.conv5_Top = convBlock(50, 50, batchNorm = True)
        self.conv6_Top = convBlock(50, 50, batchNorm = True)
        self.conv7_Top = convBlock(50, 75, batchNorm = True)
        self.conv8_Top = convBlock(75, 75, batchNorm = True)
        self.conv9_Top = convBlock(75, 75, batchNorm = True)

        self.fully_1 = nn.Conv3d(150, 400, kernel_size=1)
        self.fully_2 = nn.Conv3d(400, 100, kernel_size=1)
        self.final = nn.Conv3d(100, nClasses, kernel_size=1)
        
    def forward(self, input):

        # get the 3 channels as 5D tensors
        y_1 = self.conv1_Top(input[:,0:1,:,:,:])
        y_2 = self.conv2_Top(y_1)
        y_3 = self.conv3_Top(y_2)
        y_4 = self.conv4_Top(y_3)
        y_5 = self.conv5_Top(y_4)
        y_6 = self.conv6_Top(y_5)
        y_7 = self.conv7_Top(y_6)
        y_8 = self.conv8_Top(y_7)
        y_9 = self.conv9_Top(y_8)

        y_3_cropped = croppCenter(y_3,y_9.shape)
        y_6_cropped = croppCenter(y_6,y_9.shape)

        y = self.fully_1(torch.cat((y_3_cropped, y_6_cropped, y_9), dim=1))
        y = self.fully_2(y)
        
        return self.final(y)


class LiviaSemiDenseNet(nn.Module):
    def __init__(self, nClasses):
        super(LiviaSemiDenseNet, self).__init__()

        # Path-Top
        # self.conv1_Top = torch.nn.Conv3d(1, 25, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1_Top = convBlock(1, 25)
        self.conv2_Top = convBlock(25, 25, batchNorm=True)
        self.conv3_Top = convBlock(25, 25, batchNorm=True)
        self.conv4_Top = convBlock(25, 50, batchNorm=True)
        self.conv5_Top = convBlock(50, 50, batchNorm=True)
        self.conv6_Top = convBlock(50, 50, batchNorm=True)
        self.conv7_Top = convBlock(50, 75, batchNorm=True)
        self.conv8_Top = convBlock(75, 75, batchNorm=True)
        self.conv9_Top = convBlock(75, 75, batchNorm=True)

        self.fully_1 = nn.Conv3d(450, 400, kernel_size=1)
        self.fully_2 = nn.Conv3d(400, 100, kernel_size=1)
        self.final = nn.Conv3d(100, nClasses, kernel_size=1)

    def forward(self, input):
        # get the 3 channels as 5D tensors
        y_1 = self.conv1_Top(input[:, 0:1, :, :, :])
        y_2 = self.conv2_Top(y_1)
        y_3 = self.conv3_Top(y_2)
        y_4 = self.conv4_Top(y_3)
        y_5 = self.conv5_Top(y_4)
        y_6 = self.conv6_Top(y_5)
        y_7 = self.conv7_Top(y_6)
        y_8 = self.conv8_Top(y_7)
        y_9 = self.conv9_Top(y_8)

        y_1_cropped = croppCenter(y_1, y_9.shape)
        y_2_cropped = croppCenter(y_2, y_9.shape)
        y_3_cropped = croppCenter(y_3, y_9.shape)
        y_4_cropped = croppCenter(y_4, y_9.shape)
        y_5_cropped = croppCenter(y_5, y_9.shape)
        y_6_cropped = croppCenter(y_6, y_9.shape)
        y_7_cropped = croppCenter(y_7, y_9.shape)
        y_8_cropped = croppCenter(y_8, y_9.shape)

        y = self.fully_1(torch.cat((y_1_cropped,
                                    y_2_cropped,
                                    y_3_cropped,
                                    y_4_cropped,
                                    y_5_cropped,
                                    y_6_cropped,
                                    y_7_cropped,
                                    y_8_cropped,
                                    y_9), dim=1))
        y = self.fully_2(y)

        return self.final(y)