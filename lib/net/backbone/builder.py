# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

#import net.resnet_atrous as atrousnet
#import net.resnet as resnet
#import net.xception as xception
#import net.vgg as vgg
#import net.resnet38d as resnet38d
#import net.mobilenetv3 as mobilenetv3
#import net.mobilenetv2 as mobilenetv2
#import net.efficientnet as efficientnet
from utils.registry import BACKBONES

def build_backbone(backbone_name, pretrained=True, **kwargs):
	net = BACKBONES.get(backbone_name)(pretrained=pretrained, **kwargs)
	return net
	
#def build_backbone(backbone_name, pretrained=True, os=16):
#	if backbone_name == 'res50_atrous':
#		net = atrousnet.resnet50_atrous(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'res18_atrous':
#		net = atrousnet.resnet18_atrous(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'res26_atrous':
#		net = atrousnet.resnet26_atrous(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'res101_atrous':
#		net = atrousnet.resnet101_atrous(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'res152_atrous':
#		net = atrousnet.resnet152_atrous(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'resnet50':
#		net = resnet.resnet50(pretrained=pretrained) # os always be 8
#		return net
#	elif backbone_name == 'resnet101':
#		net = resnet.resnet101(pretrained=pretrained) # os always be 8
#		return net
#	elif backbone_name == 'eqresnet50':
#		net = eqresnet.eqresnet50(pretrained=False)
#		return net
#	elif backbone_name == 'eqresnet18':
#		net = eqresnet.eqresnet18(pretrained=False)
#		return net
#	elif backbone_name == 'xception' or backbone_name == 'Xception':
#		net = xception.xception(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'vgg16_bn' or backbone_name == 'VGG16_BN':
#		net = vgg.vgg16_bn(pretrained=pretrained)
#		return net
#	elif backbone_name == 'vgg16' or backbone_name == 'VGG16':
#		net = vgg.vgg16(pretrained=pretrained)
#		return net
#	elif backbone_name == 'resnet38' or backbone_name == 'ResNet38':
#		net = resnet38d.resnet38(pretrained=pretrained)
#		return net
#	elif backbone_name == 'mobilenetv3_super':
#		net = mobilenetv3.mobilenetv3_super(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'mobilenetv3_large':
#		net = mobilenetv3.mobilenetv3_large(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'mobilenetv3_small':
#		net = mobilenetv3.mobilenetv3_small(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'mobilenetv2':
#		net = mobilenetv2.mobilenet_v2(pretrained=pretrained, os=os)
#		return net
#	elif 'efficientnet' in backbone_name:
#		net = efficientnet.efficientnet(pretrained=pretrained, os=os, name=backbone_name)
#		return net
#	elif backbone_name == 'res18_sppm':
#		net = sppmnet.resnet18_sppm(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'res50_sppm':
#		net = sppmnet.resnet50_sppm(pretrained=pretrained, os=os)
#		return net
#	elif backbone_name == 'res101_sppm':
#		net = sppmnet.resnet101_sppm(pretrained=pretrained, os=os)
#		return net
#	else:
#		raise ValueError('backbone.py: The backbone named %s is not supported yet.'%backbone_name)
	

	

