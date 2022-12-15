import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvReluBn(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConvReluBn, self).__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1, bias = False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace = True)
  
  def forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)

    return out

class simpleCNN(nn.Module):
  def __init__(self, init_weight = True):
    super(simpleCNN, self).__init__()
    # self.block = ConvReluBn(3, 512)
    self.conv = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
    self.relu = nn.ReLU(inplace = True)
    self.block1 = ConvReluBn(64, 128)#224
    self.block2 = ConvReluBn(128, 256)#112
    self.block3 = ConvReluBn(256, 512)#56
    self.block4 = ConvReluBn(512, 512)#28
    self.maxpool = nn.MaxPool2d(2)
    self.out = nn.Sequential(
      nn.Linear(512*7*7, 128),
      nn.ReLU(inplace = True),
      nn.Linear(128,4)
    )
    if init_weight:
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
          elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
  def forward(self, x):
    out = self.conv(x)
    # print(out)
    out = self.relu(out)
    out = self.maxpool(out)
    out = self.block1(out)
    out = self.maxpool(out)
    out = self.block2(out)
    out = self.maxpool(out)
    out = self.block3(out)
    out = self.maxpool(out)
    out = self.block4(out)
    out = self.maxpool(out)
    out = torch.flatten(out, 1) 
    out = self.out(out)

    return out
    

class miniVGG(nn.Module):
  def __init__(self, init_weight = True):
    super(miniVGG, self).__init__()

    self.block1 = ConvReluBn(3,64)
    self.block2 = ConvReluBn(64,64)
    self.block3 = ConvReluBn(64,64)

    self.block4 = ConvReluBn(64,128)
    self.block5 = ConvReluBn(128,128)
 

    self.block7 = ConvReluBn(128,256)
    self.block8 = ConvReluBn(256,256)

    self.block9 = ConvReluBn(256,512)
    self.block10 = ConvReluBn(512,512)
    self.maxpool = nn.MaxPool2d(2)

    self.out = nn.Sequential(
      nn.Linear(512*7*7, 128),
      nn.ReLU(inplace = True),
      nn.Linear(128,4)
    )

    if init_weight:
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
          elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
              
  def forward(self, x):
    
    out = self.block1(x)
    out = self.block2(out)
    out = self.block3(out)
    out = self.maxpool(out) #112

    out = self.block4(out)
    out = self.block5(out)
    out = self.maxpool(out) #56

    out = self.block7(out)
    out = self.block8(out)
    out = self.maxpool(out) #28
    out = self.block9(out)
    out = self.maxpool(out)
    out = self.block10(out)
    out = self.maxpool(out)
    out = torch.flatten(out, 1) 
    
    out = self.out(out)

    return out
    

    

