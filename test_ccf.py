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
    testloader = data.DataLoader(dataset, batch_size=args.batch_size)
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
    parser.add_argument("--tocsv",default="./results/submission_101.csv", help="csv result")
    parser.add_argument("--testdir",default="/data2/hbchen/ccf/Test_Data/", help="test image dir")
    parser.add_argument("--ckpt",default="./work_space_ir_se_101_model/models/model_2019-10-10-04-15_accuracy:0.8862857142857143_step:62426_None.pth", help="model checkpoints")
    parser.add_argument("--save_mat",default='./results/face_embedding_101.mat', help="feature_mat")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-b", "--batch_size", default =100,type=int,help="batch_size")
    parser.add_argument("-ws", "--workspace", default ='work_space_multimodel',type=str)
    parser.add_argument("-s", "--struct", help="backbone struct", default='ir_se_101', type=str)
    args = parser.parse_args()

    conf = get_config(args)
    conf.struct = args.struct
    ##############################
    learner = face_learner(conf, True)
    learner.load_state(model=args.ckpt , head=None,head_race=None,optimizer=None)
    
    learner.model = learner.model.to(conf.device)
    learner.model.eval()

    print('learner loaded')
    ###############################
    data_dir = args.testdir                     # testset dir
    name_list = [name for name in os.listdir(data_dir)]
    img_paths = [data_dir + name for name in os.listdir(data_dir)]
    print('Images number:', len(img_paths))
    s = time.time()
    features = get_featurs(learner.model, img_paths)
    t = time.time() - s
    print(features.shape)
    print('total time is {}, average time is {}'.format(t, t / len(img_paths)))

    fe_dict = get_feature_dict(name_list, features)
    print('Output number:', len(fe_dict))
    sio.savemat(args.save_mat, fe_dict)
    ##################################
    face_features = sio.loadmat(args.save_mat)
    #face_features_ccf = sio.loadmat('face_embedding_test_ccf.mat')
    print('Loaded mat')
    sample_sub = open(args.csv, 'r')  # sample submission file dir
    sub = open(args.tocsv, 'w')
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
        score = '%.2f' % cosin_metric(face_features[a][0], face_features[b][0])
        sub.write(score + '\n')
        pbar.update(1)

    sample_sub.close()
    sub.close()
