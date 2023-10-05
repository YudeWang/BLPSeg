# BLPSeg

The implementation of [**BLPSeg: Balance the Label Preference in Scribble-Supervised Semantic Segmentation**](https://ieeexplore.ieee.org/abstract/document/10225696).

## Abstract

Scribble-supervised semantic segmentation is an appealing weakly supervised technique with low labeling cost. Existing approaches mainly consider diffusing the labeled region of scribble by low-level feature similarity to narrow the supervision gap between scribble labels and mask labels. In this study, we observe an annotation bias between scribble and object mask, i.e., label workers tend to scribble on the spacious region instead of corners. This label preference makes the model learn well on those frequently labeled regions but poor on rarely labeled pixels. Therefore, we propose BLPSeg to balance the label preference for complete segmentation. Specifically, the BLPSeg first predicts an annotation probability map to evaluate the rarity of labels on each image, then utilizes a novel BLP loss to balance the model training by up-weighting those rare annotations. Additionally, to further alleviate the impact of label preference, we design a local aggregation module (LAM) to propagate supervision from labeled to unlabeled regions in gradient backpropagation. We conduct extensive experiments to illustrate the effectiveness of our BLPSeg. Our single-stage method even outperforms other advanced multi-stage methods and achieves state-of-the-art performance.

## Installation

- Linux with Python 3.6
- pytorch 1.13.0, torchvision 0.14.0
- CUDA 11.7
- 2 x TITAN RTX GPUs (24G)
- `pip install -r requirements.txt`


## Getting Started

### Preparing Dataset

This repository support PASCAL VOC 2012 and PASCAL-Context dataset. The datasets are organized as follow (recommend use soft link to organize):
```
data/
	VOCdevkit/
		VOC2012/
			Annotations/
			JPEGIMages/
			ImageSets/
			SegmentationClass/
			SegmentationClassAug/
				xxxx.png
				......
			SegmentationObject/
		Context/
			ImageSets/
			JPEGImages/
			SegmentationClass/
		scribble_annotation/
			pascal_2012/
			pascal_2012_label/
			pascal_context/	
			pascal_context_label/
```

1. Download PASCAL VOC 2012 dataset following [official instruction.](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)
2. Download PASCAL VOC 2012 trainaug set (including 10582 images) from [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0), place the folder at `data/VOCdevkit/SegmentationClassAug/`.
3. Generate training list file `data/VOCdevkit/ImageSets/trainaug.txt` for trainaug set (1464 images from official VOC12 dataset + additional 9118 images determined by `data/VOCdevkit/VOC2012/SegmentationClassAug`)
```
cd data
python generate_trainauglist.py
```
4. Download PASCAL-Context dataset from [here.](https://www.cs.stanford.edu/~roozbeh/pascal-context/)
5. Download scribble annotation from [PASCAL-Scribble.](https://jifengdai.org/downloads/scribble_sup/) Convert `.xml` scribble annotation files into `.png` pixel-level annotation format
```
cd data
python xml2png_voc.py
python xml2png_context.py
```

### Train & Evaluation 

We take the experiments on PASCAL VOC 2012 as example. Firstly switch to the experiment folder.
```
cd experiment/blpseg-voc
```
Please setup the corresponding settings in `config.py` then run:
```
python train.py
```
Check the `config_dict['TEST_CKPT']` in `config.py` and run evaluation script:
```
python test.py
```
## Model Zoo

| Model | Dataset | mIoU% (w/o CRF) | Download|
|:------|:--------|------|---------|
| BLPSeg-res101 | PASCAL VOC 2012 | 77.559 | [Google Drive](https://drive.google.com/file/d/13UJZOZVIZDkdbYAhEANJks8in2sbCD93/view?usp=sharing)/[Baiduyun Drive](https://pan.baidu.com/s/1iuKk-8AgMjK78SyEOtj_ow?pwd=d9ie)(code: d9ie) |
| BLPSeg-res101 | PASCAL-Context | 45.745 | [Google Drive](https://drive.google.com/file/d/1TiVU2toU6wr1_xa6nbVuP29up_Wt4tff/view?usp=sharing)/[Baiduyun Drive](https://pan.baidu.com/s/155noxNOA9EnTZ4_6Yy01sA?pwd=pls1)(code: pls1) |

## Citations

Please cite our paper if the code is helpful to your research.

```
@article{wang2023blpseg,
  title={BLPSeg: Balance the Label Preference in Scribble-Supervised Semantic Segmentation},
  author={Wang, Yude and Zhang, Jie and Kan, Meina and Shan, Shiguang and Chen, Xilin},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```

