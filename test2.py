import json
from models.MobileNets import MobileNetV2

import torch

from models.utils import load_url, download_url

model_urls = {
    'mobilenetv2': {
        "weight": 'http://cloud.syu.life/MobilveNetV2-09-11/model.pth.tar',
	    "config": "http://cloud.syu.life/MobilveNetV2-09-11/net.config"
    }
}

res = download_url(model_urls["mobilenetv2"]["config"])
with open(res, "r") as fp:
	cfg = json.load(fp)
	net = MobileNetV2.build_from_config(cfg)
	# net.load_state_dict(load_url(model_urls["mobilenetv2"]["weight"]), strict=False)
	print(net)
	data = torch.zeros(1, 3, 224, 224)
	output = net(data)