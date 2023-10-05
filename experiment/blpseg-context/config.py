# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

config_dict = {
		'EXP_NAME': 'blpseg-context',

		'DATA_NAME': 'ContextDataset',
		'DATA_YEAR': 2012,
		'DATA_AUG': True,
		'DATA_WORKERS': 4,
		'DATA_MEAN': [0.485, 0.456, 0.406],
		'DATA_STD': [0.229, 0.224, 0.225],
		'DATA_RANDOMSCALE': [0.75, 1.25],
		'DATA_RANDOM_H': 10,
		'DATA_RANDOM_S': 10,
		'DATA_RANDOM_V': 10,
		'DATA_RANDOMCROP': 384,
		'DATA_RANDOMROTATION': 0,
		'DATA_RANDOMFLIP': 0.5,

		'MODEL_NAME': 'BLPSeg',
		'MODEL_BACKBONE': 'resnet101',
		'MODEL_BACKBONE_PRETRAIN': True,
		'MODEL_PPM_DIM': 256,
		'MODEL_NUM_CLASSES': 60,
		'MODEL_FREEZEBN': False,
		'MODEL_LAM_SIGMA': 6,

		'LOSS_GAMMA': 2.0,
		'LOSS_UNLABEL_CLASS_W': 0.02,

		'TRAIN_LR': 2.4e-5,
		'TRAIN_MOMENTUM': 0.9,
		'TRAIN_WEIGHT_DECAY': 0.01,
		'TRAIN_BN_MOM': 0.1,
		'TRAIN_POWER': 0.9,
		'TRAIN_BATCHES': 8,
		'TRAIN_SHUFFLE': False,
		'TRAIN_MINEPOCH': 0,
		'TRAIN_EPOCHS': 85,
		'TRAIN_TBLOG': True,
		'TRAIN_ST_POINT': 20000,

		'TEST_MULTISCALE': [0.5, 0.75, 1, 1.25],
		'TEST_FLIP': True,
		'TEST_CRF': False,
		'TEST_BATCHES': 1,		
}

config_dict['ROOT_DIR'] = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..'))
config_dict['MODEL_SAVE_DIR'] = os.path.join(config_dict['ROOT_DIR'],'model',config_dict['EXP_NAME'])
config_dict['TRAIN_CKPT'] = None
config_dict['LOG_DIR'] = os.path.join(config_dict['ROOT_DIR'],'log',config_dict['EXP_NAME'])
config_dict['TEST_CKPT'] = os.path.join(config_dict['ROOT_DIR'],f'model/{config_dict["EXP_NAME"]}/BLPSeg_resnet101_ContextDataset_epoch85.pth')

sys.path.insert(0, os.path.join(config_dict['ROOT_DIR'], 'lib'))
