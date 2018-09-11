import json
from models.MobileNets import MobileNetV2

with open("models/net.config", "r+") as fp:
	cfg = json.load(fp)
	net = MobileNetV2.build_from_config(cfg)
	print(net)