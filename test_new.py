import cv2
import os
import numpy as np
import time
import scipy.io as sio
from collections import OrderedDict
from tqdm import tqdm
import torch
from torch.nn import DataParallel
from torch.utils import data
from scipy.spatial.distance import pdist
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
from config import get_config
from Learner import face_learner
from torchvision import transforms as trans
from utils import load_facebank, draw_box_name, prepare_facebank
class TestSet(torch.utils.data.Dataset):
    def __init__(self,test_list):
        self.test_list = test_list
        self.transform = conf.test_transform
    def __getitem__(self,index):
        sample = self.test_list[index]
        data = Image.open(sample)
        data = self.transform(data)
        return data
    def __len__(self):
        return len(self.test_list)

def get_featurs(model, test_list):
    # device = torch.device("cuda")

    pbar = tqdm(total=len(test_list))
    dataset = TestSet(test_list=test_list)
    testloader = data.DataLoader(dataset, batch_size=1)
    with torch.no_grad():
        for idx,imgs in enumerate(testloader):
            pbar.update(imgs.shape[0])
            imgs = imgs.to(conf.device)
            if idx==0:
                feature = model(imgs)
                feature = feature.detach().cpu().numpy()
                features = feature
            else:
                feature = model(imgs)
                feature = feature.detach().cpu().numpy()
                features = np.concatenate((features, feature), axis=0)
    return features
    # for idx, img_path in enumerate(test_list):
    #     pbar.update(1)
    #     dataset = TestSet(test_list=img_path)

    #     trainloader = data.DataLoader(dataset, batch_size=1)
    #     with torch.no_grad():
    #         for img in trainloader:
    #             img = img.to(conf.device)
    #             if idx == 0:
    #                 feature = model(img)
    #                 feature = feature.detach().cpu().numpy()
    #                 features = feature
    #             else:
    #                 feature = model(img)
    #                 feature = feature.detach().cpu().numpy()
    #                 features = np.concatenate((features, feature), axis=0)
    

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def dist(x1,x2):
    diff = np.subtract(x1, x2)
    dist = np.sum(np.square(diff))
    return dist

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test for ccf dataset')
    parser.add_argument("--csv",default="/data2/hbchen/ccf/submission_template.csv", help="address for the test csv")
    parser.add_argument("--testdir",default="/data2/hbchen/ccf/Test_Data/", help="test image dir")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    args = parser.parse_args()

    conf = get_config(False)
    
    ############################################################

    ##################################
    
    face_features = sio.loadmat('face_embedding_sep.mat')
    #face_features_ccf = sio.loadmat('face_embedding_test_ccf.mat')
    print('Loaded mat')
    sample_sub = open(args.csv, 'r')  # sample submission file dir
    sub = open('submission_ccf_sep.csv', 'w')
    print('Loaded CSV')
    lines = sample_sub.readlines()
    pbar = tqdm(total=len(lines))
    for line in lines:
        pair = line.split(',')[0]
        sub.write(pair + ',')
        a, b = pair.split(':')
        
        #x1 = np.concatenate((face_features[a][0],face_features_ccf[a][0]))
        #x2 = np.concatenate((face_features[b][0],face_features_ccf[b][0]))
        # score = '%.5f' % (0.5 + 0.5 * (cosin_metric(face_features[a][0], face_features[b][0])))
        #score = '%.5f' % cosin_metric(x1,x2)
        score = '%.5f' % cosin_metric(face_features[a][0], face_features[b][0])
        sub.write(score + '\n')
        pbar.update(1)

    sample_sub.close()
    sub.close()
