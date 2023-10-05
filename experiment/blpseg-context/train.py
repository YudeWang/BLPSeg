# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import os
import sys
import cv2
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from config import config_dict
from utils.pyutils import Timer
from utils.imutils import img_denorm
from tensorboardX import SummaryWriter
from utils.finalprocess import writelog
from torch.utils.data import DataLoader
from net.generateNet import generate_net
from net.sync_batchnorm import SynchronizedBatchNorm2d
from net.sync_batchnorm.replicate import patch_replication_callback
from utils.configuration import Configuration
from datasets.generateData import generate_dataset

torch.manual_seed(1) # cpu
torch.cuda.manual_seed_all(1) #gpu
np.random.seed(1) #numpy
random.seed(1) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def train_net(cfg, comment=''):
	period = 'train'
	dataset = generate_dataset(cfg, period=period, transform='weak')
	def worker_init_fn(worker_id):
		np.random.seed(1 + worker_id)
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True,
				worker_init_fn=worker_init_fn)
	norm_layer=SynchronizedBatchNorm2d
	net = generate_net(cfg, norm_layer=norm_layer)
	net.build_distance_buffer()
	if cfg.TRAIN_CKPT:
		net.load_state_dict(torch.load(cfg.TRAIN_CKPT),strict=False)
	if cfg.TRAIN_TBLOG:
		from tensorboardX import SummaryWriter
		# Set the Tensorboard logger
		tblogger = SummaryWriter(cfg.LOG_DIR)	

	print('Use %d GPU'%torch.cuda.device_count())
	device = torch.device(0)
	if torch.cuda.device_count() > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
		module = net.module
	else:
		module = net
	net.to(device)
	optimizer = optim.AdamW(
		params = [
			{'params': module.parameters(), 'lr': cfg.TRAIN_LR},
		],
		weight_decay=cfg.TRAIN_WEIGHT_DECAY
	)
	itr = cfg.TRAIN_MINEPOCH * len(dataloader)
	max_itr = cfg.TRAIN_EPOCHS*len(dataloader)
	tblogger = SummaryWriter(cfg.LOG_DIR)
	timer = Timer("Session started: ")
	scaler = torch.cuda.amp.GradScaler()
	for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
		for i_batch, sample_batched in enumerate(dataloader):
			now_lr = adjust_lr(optimizer, itr, max_itr, cfg)
			optimizer.zero_grad()
			imgs           = sample_batched['image'].to(0)
			cls_label      = sample_batched['category'].long().to(0)
			scribble_label = sample_batched['segmentation_scribble'].to(0)
			segmentation_label = sample_batched['segmentation'].long().to(0)
			N,_,H,W = imgs.shape
			scribble_sup_size = (H//4, W//4)
			feat_size = (H//4, W//4)

			C = cls_label.shape[1]
			if itr<cfg.TRAIN_ST_POINT:
				self_training = False
			else:
				self_training = True
			with torch.cuda.amp.autocast():
				losses_list, tensor_dict = net(imgs, scribble_label, cls_label, self_training)
				loss_scrib, loss_seg = losses_list
				seg_pseudo, seg_pred, scrib_pseudo = tensor_dict

				# loss collect
				losses_dict = dict()
				losses_dict['loss_scrib'] = loss_scrib.mean()
				losses_dict['loss_seg'] = loss_seg.mean()
				loss = sum(losses_dict.values())
				losses_dict['loss_sum'] = loss
				
			if scaler is None:
				loss.backward()
				optimizer.step()
			else:
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			
			print(f'epoch:{epoch}/{cfg.TRAIN_EPOCHS}   batch:{i_batch}/{len(dataset)//cfg.TRAIN_BATCHES}   '
                              f'itr:{itr+1}   lr:{now_lr:.6f}')
			print({k:f'{v.item():.3f}' for k,v in losses_dict.items()})

			if cfg.TRAIN_TBLOG and itr%100 == 0:
				timer.update_progress((itr+1) / max_itr)
				print('Finish at: %s' % (timer.str_est_finish()))
				def vislabel(label):
					vis = label.cpu().numpy()
					color = dataset.label2colormap(vis).transpose((2,0,1))
					return color

				vis_dict = dict(
					imgs_vis = [],
					segmentation_label_vis = [],
					scribble_label_vis = [],
					scrib_pseudo_vis = [],
					seg_pseudo_vis = [],
					seg_pred_vis = [],
				)
				for i in range(N):
					vis_dict['imgs_vis'].append(img_denorm(imgs[i].detach().cpu().numpy()).astype(np.uint8))
					vis_dict['segmentation_label_vis'].append(vislabel(segmentation_label[i]))
					vis_dict['scribble_label_vis'].append(vislabel(scribble_label[i]))
					vis_dict['scrib_pseudo_vis'].append(vislabel(scrib_pseudo[i]))
					vis_dict['seg_pseudo_vis'].append(vislabel(seg_pseudo[i]))
					vis_dict['seg_pred_vis'].append(vislabel(torch.argmax(seg_pred,dim=1,keepdim=False)[i]))
				for k,v in losses_dict.items():
					tblogger.add_scalar(k, v.item(), itr)
				for k,v in vis_dict.items():
					tblogger.add_images(k, np.stack(v,axis=0), itr)

			if itr % 5000 == 0:
				save_path = os.path.join(cfg.MODEL_SAVE_DIR,
					f'{cfg.MODEL_NAME}_{cfg.MODEL_BACKBONE}_{cfg.DATA_NAME}_itr{itr}.pth')
				torch.save(module.state_dict(), save_path)
				print('%s has been saved'%save_path)

			itr += 1
		
	save_path = os.path.join(cfg.MODEL_SAVE_DIR,
		f'{cfg.MODEL_NAME}_{cfg.MODEL_BACKBONE}_{cfg.DATA_NAME}_epoch{cfg.TRAIN_EPOCHS}{comment}.pth')
	torch.save(module.state_dict(),save_path)
	print('%s has been saved'%save_path)
	if cfg.TRAIN_TBLOG:
		tblogger.close()
	writelog(cfg, period)

def adjust_lr(optimizer, itr, max_itr, cfg):
	now_lr = cfg.TRAIN_LR * (1 - itr/(max_itr+1)) ** cfg.TRAIN_POWER
	optimizer.param_groups[0]['lr'] = now_lr
	return now_lr

if __name__ == '__main__':
	cfg = Configuration(config_dict)
	train_net(cfg)


