import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def affinity_matrix(feature, no_grad=False):	
	[batch, channel, row, col] = feature.size()
	spatial = row*col
#	vector1 = feature.view(batch, channel, spatial)
#	vector2 = torch.transpose(vector1, 1, 2)
#	affinity = torch.matmul(vector2, vector1)/channel	

	vector1 = feature.view(batch, channel, spatial, 1)
	vector2 = feature.view(batch, channel, 1, spatial)
	#vector1 = vector1.expand(-1,-1,-1,spatial)
	#vector2 = vector2.expand(-1,-1,spatial,-1)
	#affinity = torch.mean(torch.pow(vector1-vector2,2),dim=1)
	affinity = torch.mean(torch.abs(vector1-vector2),dim=1)
	#affinity = torch.exp(-affinity)
	
	return affinity

def neighbor_mask(N, r):
	mask = np.zeros((N,N))
	for i in range(0,r//2+1):
		mask += np.eye(N, k=i)
	for i in range(r//2):
		for j in range(r):
			mask += np.eye(N, k=(i+1)*N-r//2+j)
	mask += mask.transpose()
	return mask

def valid_mask(cam, threshold):
	N, C, H, W = cam.size()
	max_value, _ = torch.max(cam.view(N,-1), dim=-1)
	max_value = max_value.view(N,1,1)
	max_map, _ = torch.max(cam, dim=1)
	confident = (max_map > (max_value * threshold))
	confident += (max_map <= 0)
	confident_vector = confident.view(N,1,-1).float()
	confident_mask = torch.matmul(confident_vector.transpose(1,2), confident_vector)
	return confident_mask

def invalid_mask(gt, mask):
	N, H, W = gt.size()
	invalid = torch.zeros((N,H,W))
	invalid[gt==255] = 1
	invalid = invalid.view(N,-1)
	m = mask.view(1, H*W, H*W) * invalid.view(N, H*W, 1)
	m = m * invalid.view(N, 1, H*W)
	return m

def label_expand(label_onehot, img, nb_mask):
	C, H, W = label_onehot.size()
	assert img.size()[1] == H and img.size()[2] == W
	with torch.no_grad():
		aff_img = affinity_matrix(img.unsqueeze(0)).view(1,H*W,H*W)
		aff_img_exp = torch.exp(-aff_img)*nb_mask
		aff_img_norm = aff_img_exp/(torch.sum(aff_img_exp, dim=1, keepdim=True)+1e-5)

		trans_matrix = torch.pow(aff_img_norm, 8).view(H*W, H*W)
		#trans_matrix = aff_img_norm.view(H*W, H*W)
#		for i in range(2):
#			trans_matrix = torch.matmul(trans_matrix, trans_matrix)
		result_onehot = label_onehot.view(C,-1)
		result_onehot = torch.matmul(result_onehot, trans_matrix).view(C,H,W)
		return result_onehot

def radius_mask(size, radius):
	h,w = size
	search_dist = []
	for x in range(-radius, radius+1):
		for y in range(-radius, radius+1):
			if x**2 + y**2 <= radius**2:
				search_dist.append((y,x))	
	coor_h = torch.arange(h).view(h,1).repeat(1,w).unsqueeze(-1)
	coor_w = torch.arange(w).view(1,w).repeat(h,1).unsqueeze(-1)
	coor = torch.cat([coor_h, coor_w],dim=-1)
	coor_list = []
	mask = torch.zeros((h*w,h*w))
	for dh,dw in search_dist:
		coor_now_h = coor[:,:,0]+dh
		coor_now_w = coor[:,:,1]+dw
		coor_now_h = torch.clamp(coor_now_h, 0, h-1)
		coor_now_w = torch.clamp(coor_now_w, 0, w-1)

		idx_from = coor[:,:,0] * w  + coor[:,:,1]
		idx_from = idx_from.view(-1)

		idx_to = coor_now_h * w + coor_now_w
		idx_to = idx_to.view(-1)
		mask[idx_from, idx_to] = 1	
	return mask

