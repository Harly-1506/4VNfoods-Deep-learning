import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleMLP(nn.Module):
  def __init__(self, numberOflayers, unit, init_weight = True):
    super(simpleMLP, self).__init__()

    self.layer1 = nn.Linear(224*224*3, unit)
    self.relu = nn.ReLU(inplace = True)
    addlayers = []
    for i in range(numberOflayers):
      addlayers += [ nn.Linear(unit,unit), nn.ReLU(inplace = True) ]
    
    self.out = nn.Linear(unit, 4)
    self.layers = nn.Sequential(*addlayers) 

    if init_weight:
      for m in self.modules():
          if isinstance(m, nn.Linear):
              nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
              nn.init.constant_(m.bias, 0)

  def forward(self, x):
    # print(x.size())
    x = x.view(x.size(0), -1)
    # print(x.size())
    out = self.layer1(x)
    out = self.relu(out)
    out = self.layers(out)
    out = self.out(out)

    return out
