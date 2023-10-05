import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from net.backbone import build_backbone
from utils.registry import NETS
from utils.affinity_op import radius_mask
from net.operators import PPM
from net.loss import BLPLoss

@NETS.register_module
class BLPSeg(nn.Module):
	def __init__(self, cfg, norm_layer=nn.BatchNorm2d, **kwargs):
		super(BLPSeg, self).__init__()
		self.cfg = cfg
		inplane = cfg.MODEL_PPM_DIM
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=cfg.MODEL_BACKBONE_PRETRAIN, norm_layer=norm_layer, **kwargs)
		feature_inplanes = self.backbone.num_features

		self.ppm = PPM(feature_inplanes[-1], inplane, norm_layer=norm_layer)
		self.decoder = UPerNetDecoder(feature_inplanes, inplane, norm_layer)

		self.seg_head = nn.Sequential(
				nn.Conv2d(inplane, inplane, kernel_size=1, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplane, inplane, kernel_size=1, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplane, cfg.MODEL_NUM_CLASSES, kernel_size=1, bias=True)
		)
		self.scribble_head = nn.Sequential(
				nn.Conv2d(inplane, inplane, kernel_size=1, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplane, inplane, kernel_size=1, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplane, cfg.MODEL_NUM_CLASSES+1, kernel_size=1, bias=True)
		)
		self.embed_proj = PatchEmbed(patch_size=4, in_chans=3, embed_dim=inplane)
		for m in self.modules():
			if m not in self.backbone.modules():
				if isinstance(m, (nn.Conv2d, nn.Linear)):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if isinstance(m, norm_layer):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		weight = torch.ones(cfg.MODEL_NUM_CLASSES+1)
		weight[-1] = cfg.LOSS_UNLABEL_CLASS_W
		self.CELoss = nn.CrossEntropyLoss(ignore_index=255, weight=weight)
		self.BLPLoss = BLPLoss(gamma=self.cfg.LOSS_GAMMA, ignore_index=255)

	def build_distance_buffer(self):
		self.distance_prob = torch.nn.parameter.Parameter(self.get_distance((96,96),\
								  sigma=self.cfg.MODEL_LAM_SIGMA).float().cpu(), \
								  requires_grad=False)
		
	def local_aggregation(self, shallow_feat, seg_pred):
		n,c,h,w = seg_pred.size()
		feat_prob = self.get_feat_prob(shallow_feat, (h,w), sigma=1)
		dis_prob = self.get_dis_prob((h,w), sigma=self.cfg.MODEL_LAM_SIGMA)
		sim = feat_prob * dis_prob
		sim = sim / (torch.sum(sim,dim=1,keepdim=True)+1e-7)
		seg_pred = torch.einsum('nca,nab->ncb', seg_pred.flatten(2), sim).view(n,-1,h,w)
		return seg_pred

	def forward(self, x, scribble_label=None, cls_label=None, self_training=False):
		H,W = x.size()[-2:]
		backbone_out = self.backbone(x)
		n,c,h,w = backbone_out[0].shape
		
		backbone_out.append(self.ppm(backbone_out[-1]))
		decoder_out = self.decoder(backbone_out)
		shallow_feat = self.embed_proj(x)

		# segmentation head
		seg_pred = self.seg_head(decoder_out)
		seg_pred = self.local_aggregation(shallow_feat, seg_pred)
		seg_pred = F.interpolate(seg_pred, (H,W), mode='bilinear')
		if scribble_label is None:
			return seg_pred

		tensor_collect = []
		with torch.no_grad():
			seg_prob = seg_pred.softmax(dim=1)
			seg_prob = seg_prob * cls_label
			seg_pseudo = torch.argmax(seg_prob,dim=1,keepdim=False)
			seg_pseudo_max = torch.max(seg_prob,dim=1,keepdim=False)[0]
			del seg_prob
			seg_pseudo[seg_pseudo_max<0.9] = 255
			seg_pseudo[scribble_label<self.cfg.MODEL_NUM_CLASSES] = \
				scribble_label[scribble_label<self.cfg.MODEL_NUM_CLASSES]
			seg_pseudo[scribble_label==255] = 255
			tensor_collect.append(seg_pseudo)
			tensor_collect.append(seg_pred)

		# scribble head
		scrib_pred = self.scribble_head(decoder_out.detach())
		scrib_pred = F.interpolate(scrib_pred, (H,W), mode='bilinear')
		with torch.no_grad():
			scrib_pseudo = scrib_pred.clone()
			scrib_pseudo[:,:-1] = scrib_pseudo[:,:-1]*cls_label
			scrib_pseudo = torch.argmax(scrib_pseudo,dim=1,keepdim=False)
			scrib_pseudo[scribble_label<self.cfg.MODEL_NUM_CLASSES] = \
				scribble_label[scribble_label<self.cfg.MODEL_NUM_CLASSES]
			scrib_pseudo[scrib_pseudo>=self.cfg.MODEL_NUM_CLASSES] = 255
		tensor_collect.append(scrib_pseudo)

		# loss
		loss_collect = []
		loss_scrib = self.CELoss(scrib_pred, scribble_label)
		loss_collect.append(loss_scrib)
		
		if self_training:
			sup = seg_pseudo
		else:
			sup = scribble_label
		sup_onehot = label2onehot(sup, ignore_index=255, num_classes=self.cfg.MODEL_NUM_CLASSES+1)[:,:-1]
		loss_seg = self.BLPLoss(seg_pred, scrib_pred.detach(), sup_onehot, cls_label)
		loss_collect.append(loss_seg)

		return loss_collect, tensor_collect

	@torch.no_grad()
	def get_dis_prob(self, size, sigma=6):
		if self.training:
			return self.distance_prob.detach()
		else:
			distance_prob = self.get_distance(size,sigma).float().cuda()
			return distance_prob.detach()

	@torch.no_grad()
	def get_color_prob(self, img, size, sigma=0.5):
		with torch.no_grad():
			if(img.shape[-2] == size[0] and img.shape[-1] == size[1]):
				img = img.flatten(2)
			else:
				img = F.interpolate(img,size,mode='bilinear').flatten(2)
			delta_color = torch.sum((img.unsqueeze(2) - img.unsqueeze(-1))**2,dim=1,keepdim=False)
			del img
			sim_color = torch.exp(-delta_color/(2 * sigma**2))
			del delta_color
			return sim_color

	def get_feat_prob(self, feat, size, sigma=1):
		if(feat.shape[-2] == size[0] and feat.shape[-1] == size[1]):
			feat = feat.flatten(2)
		else:
			feat = F.interpolate(feat,size,mode='bilinear').flatten(2)
		feat_sim = torch.einsum('nca,ncb->nab',feat,feat).softmax(dim=1)
		return feat_sim

	#@torch.no_grad()
	#def get_distance(self, size, sigma=6):
	#	h,w = size
	#	grid_h, grid_w = torch.meshgrid(torch.arange(h).cuda(), torch.arange(w).cuda())
	#	coord = torch.stack([grid_h, grid_w]) #2 x H x W
	#	del grid_h, grid_w
	#	coord1 = coord.flatten(1).unsqueeze(-1).repeat(1,1,h*w)
	#	del coord
	#	coord2 = coord1.transpose(1,2)
	#	delta = torch.sum((coord1 - coord2)**2, dim=0, keepdim=False) # HW x HW
	#	del coord1, coord2
	#	delta = torch.exp(-delta/(2*sigma**2))
	#	return delta.unsqueeze(0)

	@torch.no_grad()
	def get_distance(self, size, sigma=6):
		h,w = size
		grid_h, grid_w = torch.meshgrid(torch.arange(h).cuda(), torch.arange(w).cuda())
		coord = torch.stack([grid_h, grid_w]).flatten(1).transpose(0,1).contiguous().float()#HW x 2
		del grid_h, grid_w
		delta = torch.cdist(coord, coord, p=2) ** 2
		del coord
		delta = torch.exp(-delta/(2*sigma**2))
		return delta.unsqueeze(0)


class UPerNetDecoder(nn.Module):
	def __init__(self, feature_inplanes, inplane, norm_layer):
		super().__init__()
		self.lateral_convs = nn.ModuleList()
		self.fpn_convs = nn.ModuleList()
		for i in range(3):
			self.lateral_convs.append(nn.Sequential(
				nn.Conv2d(feature_inplanes[i], inplane, kernel_size=1, bias=False),
				norm_layer(inplane),
				nn.ReLU(inplace=True)
			))
			self.fpn_convs.append(nn.Sequential(
				nn.Conv2d(inplane, inplane, kernel_size=3, padding=1, bias=False),
				norm_layer(inplane),
				nn.ReLU(inplace=True)
			))
			
		self.fpn_bottleneck = nn.Sequential(
				nn.Conv2d(inplane*4, inplane, kernel_size=3, padding=1, bias=False),
				norm_layer(inplane),
				nn.ReLU(inplace=True)
		)
		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Linear)):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			if isinstance(m, norm_layer):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, conv_out):
		"""Forward function."""
		laterals = [
			lateral_conv(conv_out[i])
			for i, lateral_conv in enumerate(self.lateral_convs)
		]
		laterals.append(conv_out[-1])
		
		# build top-down path
		used_backbone_levels = len(laterals)
		for i in range(used_backbone_levels - 1, 0, -1):
			prev_shape = laterals[i - 1].shape[2:]
			laterals[i - 1] = laterals[i-1] + F.interpolate(
				laterals[i],
				size=prev_shape,
				mode='bilinear',
				align_corners=False)

		# build outputs
		fpn_outs = [
			self.fpn_convs[i](laterals[i])
			for i in range(used_backbone_levels - 1)
		]
		# append psp feature
		fpn_outs.append(laterals[-1])

		for i in range(used_backbone_levels - 1, 0, -1):
			fpn_outs[i] = F.interpolate(
				fpn_outs[i],
				size=fpn_outs[0].shape[2:],
				mode='bilinear',
				align_corners=False)
		fpn_outs = torch.cat(fpn_outs, dim=1)
		output = self.fpn_bottleneck(fpn_outs)
		return output

class PatchEmbed(nn.Module):
	""" Image to Patch Embedding from swin transformer
	Args:
		patch_size (int): Patch token size. Default: 4.
		in_chans (int): Number of input image channels. Default: 3.
		embed_dim (int): Number of linear projection output channels. Default: 96.
		norm_layer (nn.Module, optional): Normalization layer. Default: None
	"""

	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
		super().__init__()
		self.patch_size = (patch_size, patch_size)
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
		if norm_layer is not None:
			self.norm = norm_layer(embed_dim)
		else:
			self.norm = None
		self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)

	def forward(self, x):
		"""Forward function."""
		# padding
		_, _, H, W = x.size()
		if W % self.patch_size[1] != 0:
			x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
		if H % self.patch_size[0] != 0:
			x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

		x = self.proj(x)  # B C Wh Ww
		if self.norm is not None:
			Wh, Ww = x.size(2), x.size(3)
			x = x.flatten(2).transpose(1, 2)
			x = self.norm(x)
			x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
		x = self.conv(x)
		return x

@torch.no_grad()	
def label2onehot(gt_semantic_seg, ignore_index=None, num_classes=None):
	seg = gt_semantic_seg.clone()
	if seg.dim() == 4:
		seg = seg.squeeze(1)
	n,h,w = seg.shape
	if num_classes is not None:
		max_value = num_classes
	else:
		max_value = torch.max(seg)+1
	if ignore_index is not None:
		seg[seg == ignore_index] = max_value
	gt_onehot = F.one_hot(seg.long(), max_value+1).permute(0,3,1,2).contiguous().float()
	return gt_onehot
