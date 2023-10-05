# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os
import cv2
import torch
import multiprocessing
import numpy as np
import pandas as pd
import scipy.io as scio
from PIL import Image
from skimage import io
from utils.registry import DATASETS
from torch.utils.data import Dataset
from datasets.BaseDataset import BaseDataset
from utils.imutils import *
from datasets.transform import *

@DATASETS.register_module
class ContextDataset(BaseDataset):
	def __init__(self, cfg, period, transform='none'):
		super(ContextDataset,self).__init__(cfg, period, transform)
		self.dataset_name = 'Context'
		self.root_dir = os.path.join(cfg.ROOT_DIR,'data','VOCdevkit')
		self.dataset_dir = os.path.join(self.root_dir,self.dataset_name)
		self.rst_dir = os.path.join(self.root_dir,'results',self.dataset_name,'Segmentation')
		self.eval_dir = os.path.join(self.root_dir,'eval_result',self.dataset_name,'Segmentation')
		self.img_dir = os.path.join(self.dataset_dir, 'JPEGImages')
		self.seg_dir = os.path.join(self.dataset_dir, 'SegmentationClass')
		self.set_dir = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation')
		self.scribble_gt_dir = os.path.join(self.root_dir, 'scribble_annotation', 'pascal_context_label')
		self.pseudo_dir = None

		file_name = self.set_dir+'/'+period+'.txt'
		df = pd.read_csv(file_name, names=['filename'])
		self.name_list = df['filename'].values
	
		self.label_mapping, self.categories = self.__labelmapping() 
		self.categories = list(self.categories)
		self.num_categories = len(self.categories)+1
		assert self.num_categories == self.cfg.MODEL_NUM_CLASSES
		self.cmap = self.__colormap(self.num_categories)

	def __len__(self):
		return len(self.name_list)

	def set_pseudo_dir(self, pseudo_dir):
		self.pseudo_dir = pseudo_dir

	def load_name(self, idx):
		name = self.name_list[idx]
		return name

	def load_image(self, idx):
		name = self.name_list[idx]
		img_file = self.img_dir + '/' + name + '.jpg'
		image = cv2.imread(img_file)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image_rgb

	def load_segmentation(self, idx):
		name = self.name_list[idx]
		seg_file = self.seg_dir + '/' + name + '.mat'
		segmentation = scio.loadmat(seg_file)['LabelMap']
		for i in range(len(self.label_mapping)):
			segmentation[segmentation==i] = self.label_mapping[i] 
		segmentation[segmentation>len(self.categories)] = 0
		return segmentation

	def load_pseudo(self, idx):
		name = self.name_list[idx]
		seg_file = self.pseudo_dir + '/' + name + '.png'
		segmentation = np.array(Image.open(seg_file))
		return segmentation

	def load_scribble(self, idx):
		name = self.name_list[idx]
		seg_file = self.scribble_gt_dir + '/' + name + '.png'
		scribble = np.array(Image.open(seg_file))
		scribble[scribble==255] = self.num_categories
		return scribble

	def __sample_generate__(self, idx, split_idx=0):
		name = self.load_name(idx)
		image = self.load_image(idx)
		r,c,_ = image.shape
		sample = {'image': image, 'name': name, 'row': r, 'col': c, 'image_orig': image}

		if 'test' in self.period:
			return self.__transform__(sample)
		else:
			segmentation = self.load_segmentation(idx)
		sample['segmentation'] = segmentation
		if self.pseudo_dir is not None and 'train' in self.period and self.transform != 'none':
			sample['segmentation_pseudo'] = self.load_pseudo(idx)
		if self.scribble_gt_dir is not None and 'train' in self.period:
			sample['segmentation_scribble'] = self.load_scribble(idx)
			t = sample['segmentation_scribble'].copy()
			t[t >= self.num_categories] = 0
			sample['category'] = seg2cls(t,self.num_categories)

		sample = self.__transform__(sample)
		return sample
		
	def __labelmapping(self):
		path1 = os.path.join(self.seg_dir, 'labels.txt')
		path2 = os.path.join(self.seg_dir, '59_labels.txt')
		index_label1 = ['']
		index_label2 = {}
		file1 = open(path1)
		line = file1.readline()
		while line:
			line = line.replace('\n','')
			s = line.split(': ')
			index_label1.append(s[1])
			line = file1.readline()
		file1.close()
		file2 = open(path2)
		line = file2.readline()
		while line:
			line = line.replace('\n','')
			s = line.split(': ')
			index_label2[s[1]]=int(s[0])
			line = file2.readline()
		file2.close()
		index_mapping_list = []
		for cls in index_label1:
			if cls in index_label2.keys():
				index_mapping_list.append(index_label2[cls])
			else:
				index_mapping_list.append(0)
		return index_mapping_list, index_label2.keys() 


	def __colormap(self, N):
		"""Get the map from label index to color

		Args:
			N: number of class

			return: a Nx3 matrix

		"""
		cmap = np.zeros((N, 3), dtype = np.uint8)

		def uint82bin(n, count=8):
			"""returns the binary of integer n, count refers to amount of bits"""
			return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

		for i in range(N):
			r = 0
			g = 0
			b = 0
			idx = i
			for j in range(7):
				str_id = uint82bin(idx)
				r = r ^ ( np.uint8(str_id[-1]) << (7-j))
				g = g ^ ( np.uint8(str_id[-2]) << (7-j))
				b = b ^ ( np.uint8(str_id[-3]) << (7-j))
				idx = idx >> 3
			cmap[i, 0] = r
			cmap[i, 1] = g
			cmap[i, 2] = b
		return cmap
	
	def label2colormap(self, label):
		m = label.astype(np.uint8)
		r,c = m.shape
		cmap = np.zeros((r,c,3), dtype=np.uint8)
		cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
		cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
		cmap[:,:,2] = (m&4)<<5
		return cmap
	
	def save_result(self, result_list, model_id, comment=''):
		"""Save test results

		Args:
			result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

		"""
		i = 1
		#folder_path = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
		folder_path = os.path.join(self.rst_dir,f'{model_id}_{self.period}_cls{comment}')
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		for sample in result_list:
			file_path = os.path.join(folder_path, '%s.png'%sample['name'])
			cv2.imwrite(file_path, sample['predict'])
			#print('[%d/%d] %s saved'%(i,len(result_list),file_path))
			i+=1

	def save_pseudo(self, result_list):
		assert self.pseudo_dir is not None
		folder_path = self.pseudo_dir
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
			
		for sample in result_list:
			file_path = os.path.join(folder_path, '%s.png'%sample['name'])
			cv2.imwrite(file_path, sample['predict'])

	def do_python_eval(self, model_id,comment=''):
		predict_folder = os.path.join(self.rst_dir,f'{model_id}_{self.period}_cls{comment}')
		gt_folder = self.seg_dir
		TP = []
		P = []
		T = []
		for i in range(self.cfg.MODEL_NUM_CLASSES):
			TP.append(multiprocessing.Value('i', 0, lock=True))
			P.append(multiprocessing.Value('i', 0, lock=True))
			T.append(multiprocessing.Value('i', 0, lock=True))
		
		def compare(start,step,TP,P,T):
			for idx in range(start,len(self.name_list),step):
				#print('%d/%d'%(idx,len(self.name_list)))
				name = self.name_list[idx]
				predict_file = os.path.join(predict_folder,'%s.png'%name)
				gt_file = os.path.join(gt_folder,'%s.mat'%name)
				predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
				#gt = np.array(Image.open(gt_file))
				gt = scio.loadmat(gt_file)['LabelMap']
				for i in range(len(self.label_mapping)):
					gt[gt==i] = self.label_mapping[i] 
				gt[gt>len(self.categories)] = 0
				cal = gt<255
				mask = (predict==gt) * cal
		  
				for i in range(self.cfg.MODEL_NUM_CLASSES):
					P[i].acquire()
					P[i].value += np.sum((predict==i)*cal)
					P[i].release()
					T[i].acquire()
					T[i].value += np.sum((gt==i)*cal)
					T[i].release()
					TP[i].acquire()
					TP[i].value += np.sum((gt==i)*mask)
					TP[i].release()
		p_list = []
		for i in range(8):
			p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T))
			p.start()
			p_list.append(p)
		for p in p_list:
			p.join()
		IoU = []
		for i in range(self.cfg.MODEL_NUM_CLASSES):
			IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
		loglist = {}
		for i in range(self.cfg.MODEL_NUM_CLASSES):
			if i == 0:
				print('%11s:%7.3f%%'%('background',IoU[i]*100),end='\t')
				loglist['background'] = IoU[i]*100
			else:
				if i%2 != 1:
					print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100),end='\t')
				else:
					print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100))
				loglist[self.categories[i-1]] = IoU[i]*100
					
		miou = np.mean(np.array(IoU))
		print('\n======================================================')
		print('%11s:%7.3f%%'%('mIoU',miou*100))	
		loglist['mIoU'] = miou*100
		return loglist

