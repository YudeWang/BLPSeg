# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
from utils.registry import NETS

def generate_net(cfg, **kwargs):
	net = NETS.get(cfg.MODEL_NAME)(cfg, **kwargs)
	return net
