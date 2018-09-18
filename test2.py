import json

import torch

from models.MobileNets import mobilenet_v2, MobilveNetv2_wrapper as mnet
from models.resnet import resnet50


from models.models import Resnet, ResnetDilated
#
# n1 = resnet50()
# n11 = Resnet(n1)

n2 = mobilenet_v2()
device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
n2.to(device)
# n22 = mnet(n2)
#
# data = torch.zeros(1, 3, 600, 454)
#
# o1 = n11(data)
# o2 = n22(data)
#
# [print(_.size()) for _ in o1]
# print("==============================================================")
# [print(_.size()) for _ in o2]
#
#
# print(n2)

# dummy_input = torch.zeros(1, 3, 224, 224)
# torch.onnx.export(n2, dummy_input,"export.onnx", verbose=True, )

dummy = torch.randn(1, 3, 224, 224).to(device)

# warm up
import time
start = time.time()
for i in range(10):
	out = n2(dummy)
end = time.time()
print("duration %.4f " % (end - start))


start = time.time()
for i in range(10):
	out = n2(dummy)
end = time.time()
print("duration %.4f" %  (end - start))
