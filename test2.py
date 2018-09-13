import json

import torch

from models.MobileNets import mobilenet_v2
from models.resnet import resnet50


from models.models import Resnet, ResnetDilated
from models.models import MobilveNetv2 as mnet


n1 = resnet50()
n11 = Resnet(n1)

n2 = mobilenet_v2()
n22 = mnet(n2)

data = torch.zeros(1, 3, 600, 454)

o1 = n11(data)
o2 = n22(data)

[print(_.size()) for _ in o1]
print("==============================================================")
[print(_.size()) for _ in o2]