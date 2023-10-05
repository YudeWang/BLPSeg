import os
import argparse
import pandas as pd

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--list_folder', type=str, default='./VOCdevkit/VOC2012/ImageSets/Segmentation')
	parser.add_argument('--aug_folder', type=str, default='./VOCdevkit/VOC2012/SegmentationClassAug')
	args = parser.parse_args()

	train_file = os.path.join(args.list_folder, 'train.txt')
	val_file = os.path.join(args.list_folder, 'val.txt')
	trainaug_file = os.path.join(args.list_folder, 'trainaug.txt')
	train_list = pd.read_csv(train_file, names=['filename'])['filename'].values
	val_list = pd.read_csv(val_file, names=['filename'])['filename'].values
	files = os.listdir(args.aug_folder)
	trainaug_txt_file = open(trainaug_file, 'w')
	for f in files:
		fname = f[:-4]
		if fname not in val_list:
			trainaug_txt_file.write(f[:-4]+'\n')
	trainaug_txt_file.close()
	
