import torch
import torch.nn as nn
from model.resnet import res
import torch.nn.functional as F
class Dil(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Dil, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.conv_cat=nn.Conv2d(out_channel*3,out_channel,3,padding=1)
        self.conv_res=nn.Conv2d(in_channel,out_channel,1)
    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = torch.cat((x0,x1,x2),1)
        x_cat = self.conv_cat(x_cat)
        y=self.relu(x_cat+self.conv_res(x))

        return y

class Attention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Attention,self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.prelu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3,padding=1)
    def forward(self, x, y):
        x1 = x
        y1 = y
        if y1==None:
            f_cat=x1
            x1 = self.conv0(x1)
        else:
            f_cat = torch.cat((x1,y1),1)
        f_channel = self.sigmoid(self.conv2(self.prelu(self.conv1(self.global_pool(f_cat)))))
        f_channel = f_channel*x1

        max_out, _ = torch.max(f_channel, dim=1, keepdim=True)
        f_spatial = self.sigmoid(self.conv3(max_out))
        f_out = f_channel*f_spatial
        if y1==None:
            out=f_out
        else:
            out=x+f_out
        return out
class SPLIT(nn.Module):
    def __init__(self):
        super(SPLIT, self).__init__()
        self.conv1 = nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(128, 128 // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(128 // 16, 128, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x1 = self.conv1(x)
        (y1,y2,y3,y4) = torch.split(x1,x1.size()[1]//4,dim=1)
        y = y1+y2+y3+y4
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(y))))
        z = self.sigmoid(max_out)
        out = y1*z+y2*z+y3*z+y4*z
        return out
class DFINet(nn.Module):#输入三通道
    def __init__(self, in_channels):
        super(DFINet, self).__init__()
        resnet = res(depth=34)
        # ************************* Encoder ***************************
        # input conv3*3,64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#计算得到112.5 但取112 向下取整
        # Extract Features
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # Bridge
        self.convbg_1 = nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1)
        self.bnbg_1 = nn.BatchNorm2d(128)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(128, 128, kernel_size=3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(128)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(128, 128, kernel_size=3, dilation=4, padding=4)
        self.bnbg_2 = nn.BatchNorm2d(128)
        self.relubg_2 = nn.ReLU(inplace=True)

        #upsample
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        #downsample
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')
        #dil
        #self.Di5 = Dil(128,128)
        self.Di4 = Dil(512,128)
        self.Di3 = Dil(256,128)
        self.Di2 = Dil(128,128)
        self.Di1 = Dil(64,128)
        self.Di0 = Dil(64,128)
        #H1
        self.H10 = Attention(128,128)
        self.H11 = Attention(256,128)
        self.H12 = Attention(256,128)
        self.H13 = Attention(256,128)
        self.H14 = Attention(256,128)
        #
        self.conv1_0_1 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0))
        self.conv1_0_2 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1))
        self.bn1_0 = nn.BatchNorm2d(128)
        self.conv1_1_1 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0))
        self.conv1_1_2 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1))
        self.bn1_1 = nn.BatchNorm2d(128)
        self.conv1_2_1 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0))
        self.conv1_2_2 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1))
        self.bn1_2 = nn.BatchNorm2d(128)
        self.conv1_3_1 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0))
        self.conv1_3_2 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1))
        self.bn1_3 = nn.BatchNorm2d(128)
        self.conv1_4_1 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0))
        self.conv1_4_2 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1))
        self.bn1_4 = nn.BatchNorm2d(128)

        #H2
        self.conv2_0_1 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0))
        self.conv2_0_2 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1))
        self.bn2_0 = nn.BatchNorm2d(128)
        self.conv2_1_1 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0))
        self.conv2_1_2 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1))
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2_1 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0))
        self.conv2_2_2 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1))
        self.bn2_2 = nn.BatchNorm2d(128)
        self.conv2_3_1 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0))
        self.conv2_3_2 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1))
        self.bn2_3 = nn.BatchNorm2d(128)
        self.conv2_4_1 = nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0))
        self.conv2_4_2 = nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1))
        self.bn2_4 = nn.BatchNorm2d(128)

        self.sigmoid = nn.Sigmoid()


        self.conv_0 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_1 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_4 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        
        self.SP4 = SPLIT()
        self.SP3 = SPLIT()
        self.SP2 = SPLIT()
        self.SP1 = SPLIT()
        self.SP0 = SPLIT()
        
        self.convj3 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.convj2 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.convj1 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.convj0 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        #OUT
        self.conv_s0 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_s1 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_s2 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_s3 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_s4 = nn.Conv2d(128,1,kernel_size=1)

        self.conv_out1 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.bn_out1 = nn.BatchNorm2d(128)
        self.relu_out1 = nn.ReLU(inplace=True)
        self.conv_out2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.bn_out2 = nn.BatchNorm2d(128)
        self.relu_out2 = nn.ReLU(inplace=True)

        self.conv_sal = nn.Conv2d(128,1,kernel_size=1)
    def forward(self, x):
        # ************************* Encoder ***************************
        # input
        tx = self.conv1(x)
        tx = self.bn1(tx)
        f0 = self.relu(tx)
        tx = self.maxpool(f0)
        # Extract Features
        f1 = self.encoder1(tx)
        f2 = self.encoder2(f1)
        f3 = self.encoder3(f2)
        f4 = self.encoder4(f3)
        # Bridge
        tx = self.relubg_1(self.bnbg_1(self.convbg_1(f4)))
        tx = self.relubg_m(self.bnbg_m(self.convbg_m(tx)))
        f5 = self.relubg_2(self.bnbg_2(self.convbg_2(tx)))
        #dil
        D5 = f5
        D4 = self.Di4(f4)
        D3 = self.Di3(f3)
        D2 = self.Di2(f2)
        D1 = self.Di1(f1)
        D0 = self.Di0(f0)

        #**********第一部分*************#
        #H1
        h10 = self.H10(D0,None)
        h10 = self.relu(self.bn1_0(self.conv1_0_2(self.conv1_0_1(h10))))

        h11 = self.H11(D1,(self.downsample(h10)))
        h11 = self.relu(self.bn1_1(self.conv1_1_2(self.conv1_1_1(h11))))

        h12 = self.H12(D2,(self.downsample(h11)))
        h12 = self.relu(self.bn1_2(self.conv1_2_2(self.conv1_2_1(h12))))

        h13 = self.H13(D3,(self.downsample(h12)))
        h13 = self.relu(self.bn1_3(self.conv1_3_2(self.conv1_3_1(h13))))

        h14 = self.H14(D4,(self.downsample(h13)))
        h14 = self.relu(self.bn1_4(self.conv1_4_2(self.conv1_4_1(h14))))
        
        h24 = self.SP4(torch.cat((D4,self.upsample1(D5)),1))
        h24 = self.relu(self.bn2_4(self.conv2_4_2(self.conv2_4_1(h24))))

        h23 = self.SP3(torch.cat((D3,self.upsample1(h24)),1))
        h23 = self.relu(self.bn2_3(self.conv2_3_2(self.conv2_3_1(h23))))

        h22 = self.SP2(torch.cat((D2,self.upsample1(h23)),1))
        h22 = self.relu(self.bn2_2(self.conv2_2_2(self.conv2_2_1(h22))))

        h21 = self.SP1(torch.cat((D1,self.upsample1(h22)),1))
        h21 = self.relu(self.bn2_1(self.conv2_1_2(self.conv2_1_1(h21))))

        h20 = self.SP0(torch.cat((D0,self.upsample1(h21)),1))
        h20 = self.relu(self.bn2_0(self.conv2_0_2(self.conv2_0_1(h20))))

        s0 = self.conv_0(torch.cat((h10,h20),1))
        s1 = self.conv_1(torch.cat((h11,h21),1))
        s2 = self.conv_2(torch.cat((h12,h22),1))
        s3 = self.conv_3(torch.cat((h13,h23),1))
        s4 = self.conv_4(torch.cat((h14,h24),1))

        F4 = s4
        F3 = self.convj3(torch.cat((s3,self.upsample1(F4)),1))
        F2 = self.convj2(torch.cat((s2,self.upsample1(F3)),1))
        F1 = self.convj1(torch.cat((s1,self.upsample1(F2)),1))
        F0 = self.convj0(torch.cat((s0,self.upsample1(F1)),1))
        #OUT
        sal0 = self.conv_s0(F0)
        sal1 = self.upsample1(self.conv_s1(F1))
        sal2 = self.upsample2(self.conv_s2(F2))
        sal3 = self.upsample3(self.conv_s3(F3))
        sal4 = self.upsample4(self.conv_s4(F4))

        sal_out1 = self.relu_out1(self.bn_out1(self.conv_out1(F0)))
        sal_out2 = self.relu_out2(self.bn_out2(self.conv_out2(sal_out1)))
        sal_out = self.conv_sal(sal_out2)
        return F.sigmoid(sal_out),F.sigmoid(sal0),F.sigmoid(sal1),F.sigmoid(sal2),F.sigmoid(sal3),F.sigmoid(sal4)