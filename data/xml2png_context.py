import os
import argparse
import numpy as np
import xml.dom.minidom as minidom
from xml.dom.minidom import parse
from PIL import Image
from tqdm import tqdm
import time

def xml2dict(xml_file):
    result = {}
    tree = minidom.parse(xml_file)
    collection = tree.documentElement
    size = collection.getElementsByTagName('size')[0]
    h = int(size.getElementsByTagName('height')[0].childNodes[0].data)
    w = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    result['size'] = (h,w)
    result['filename'] = collection.getElementsByTagName('filename')[0].childNodes[0].data
    polygons = collection.getElementsByTagName('polygon')
    polygon_list = []
    for polygon in polygons:
        single_polygon_dict = {}
        single_polygon_dict['category'] = polygon.getElementsByTagName('tag')[0].childNodes[0].data
        points = polygon.getElementsByTagName('point')
        point_list = []
        for point in points:
            x = int(point.getElementsByTagName('X')[0].childNodes[0].data)
            y = int(point.getElementsByTagName('Y')[0].childNodes[0].data)
            x = max(min(x,w-1),0)
            y = max(min(y,h-1),0)
            point_list.append((y,x))
        single_polygon_dict['points'] = point_list
        polygon_list.append(single_polygon_dict)
    result['polygons'] = polygon_list
    return result

def drawline(img, pos1, pos2, value):
    r1,c1 = pos1
    r2,c2 = pos2
    m = max(np.abs(r1-r2), np.abs(c1-c2))
    if m <= 1:
        return img
    delta_r = (r2-r1)/m
    delta_c = (c2-c1)/m
    for i in range(m):
        r = int(r1 + delta_r*i)
        c = int(c1 + delta_c*i)
        img[r,c] = value
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=str, default='./VOCdevkit/scribble_annotation/pascal_context')
    parser.add_argument('--save', type=str, default='./VOCdevkit/scribble_annotation/pascal_context_label')
    args = parser.parse_args()

    cls2idx = {'background':0, 'plane': 1, 'bike': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 
                'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'table': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 
                'person': 15, 'plant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'monitor': 20, 'bag': 21, 'bed': 22, 
                'bench': 23, 'book': 24, 'building': 25, 'cabinet': 26, 'ceiling': 27, 'cloth': 28, 'computer': 29, 
                'cup': 30, 'door': 31, 'fence': 32, 'floor': 33, 'flower': 34, 'food': 35, 'grass': 36, 'ground': 37, 
                'keyboard': 38, 'light': 39, 'mountain': 40, 'mouse': 41, 'curtain': 42, 'platform': 43, 'sign': 44, 
                'plate': 45, 'road': 46, 'rock': 47, 'shelves': 48, 'sidewalk': 49, 'sky': 50, 'snow': 51, 'bedclothes': 52, 
                'track': 53, 'tree': 54, 'truck': 55, 'wall': 56, 'water': 57, 'window': 58, 'wood': 59}
    g = os.walk(args.xml)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    for path, dir_list, file_list in g:
        with tqdm(total=len(file_list)) as pbar:
            pbar.set_description('Processing:')
            for file_name in file_list:
                filename = os.path.join(path, file_name)
                info = xml2dict(filename)
                label = np.ones(info['size'])*255
                for polygon in info['polygons']:
                    clsidx = cls2idx[polygon['category']]
                    for i in range(len(polygon['points'])-1):
                        point1 = polygon['points'][i]
                        point2 = polygon['points'][i+1]
                        label = drawline(label, point1, point2, clsidx)
                        label[point1] = clsidx
                        label[point2] = clsidx
                label = label.astype(np.uint8)
                label = Image.fromarray(label)
                out_name = os.path.join(args.save, file_name.replace('.xml','.png'))
                label.save(out_name)
                time.sleep(0.01)
                pbar.update(1)
