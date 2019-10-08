from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader,WeightedRandomSampler
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import random
import bisect

def de_preprocess(tensor):
    return tensor*0.5 + 0.5
    
def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.Resize((112,112)),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

class ccf_test_dataset(Dataset):
    def __init__(self,imgs_folder):
        self.ds,self.class_num = self.get_ccf_dataset(imgs_folder)
        for i,sub_folder in enumerate(ds):
            sub_data = np.array(sub_folder.imgs)
            if i==0:
                self.data=sub_data
            else:
                sub_data[:,1] += class_num[i-1]
                self.data = np.concatenate((self.data,sub_data),axis=0)
        assert self.data[-1][1]==np.sum(self.class_num)-1     

    def __getitem__(self,index):
        return self.data[index]
    def __len__(self):
        length=0
        for i in self.ds:
            length += len(i)
        return length
    def class_num(self):
        return self.class_num

    def get_ccf_dataset(self,imgs_folder):
        train_transform = trans.Compose([
            trans.Resize((112,112)),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        ds = []
        class_num = []
        for path in imgs_folder.iterdir():
            if path.is_file():
                continue
            else:
                ds.append(ImageFolder(path,train_transform))
                class_num.append(ds[path][-1][1]+1)
        return ds, class_num

def get_train_loader(conf):
    if conf.data_mode in ['ms1m', 'concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder/'imgs')
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'concat']:
        vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder/'imgs')
        print('vgg loader generated')        
    if conf.data_mode == 'vgg':
        ds = vgg_ds
        class_num = vgg_class_num
    elif conf.data_mode == 'ms1m':
        ds = ms1m_ds
        class_num = ms1m_class_num
    elif conf.data_mode == 'concat':
        for i,(url,label) in enumerate(vgg_ds.imgs):
            vgg_ds.imgs[i] = (url, label + ms1m_class_num)
        ds = ConcatDataset([ms1m_ds,vgg_ds])
        class_num = vgg_class_num + ms1m_class_num
    elif conf.data_mode == 'emore':
        ds, class_num = get_train_dataset(conf.emore_folder/'imgs')
    elif conf.data_mode == 'African':
        ds, class_num = get_train_dataset(conf.ccf_folder/'African')
    elif conf.data_mode == 'Caucasian':
        ds, class_num = get_train_dataset(conf.ccf_folder/'Caucasian')
    elif conf.data_mode == 'Asian':
        ds, class_num = get_train_dataset(conf.ccf_folder/'Asian')
    elif conf.data_mode == 'Indian':
        ds, class_num = get_train_dataset(conf.ccf_folder/'Indian')
    elif conf.data_mode =='ccf':
        ds = []
        class_num = []
        imgs_num = []
        for path in conf.ccf_folder.iterdir():
            if path.is_file():
                continue
            else:
                ds_tmp, class_num_tmp = get_train_dataset(path)
                ds.append(ds_tmp)
                class_num.append(class_num_tmp)
                imgs_num.append(len(ds_tmp))
        for j, sub_ds in enumerate(ds):
            for i, (url, label) in enumerate(sub_ds.imgs):
                if j>0:
                    sub_ds.imgs[i] = (url, label + sum(class_num[:j]))
        ds = ConcatDataset(ds)
        # ds = ccf_test_dataset(conf.ccf_folder)
        # class_num = ds.class_num()
    elif conf.data_mode == 'pair_wise':
        ds, class_num, imgs_num = get_pair_ranking_dataset(conf, conf.ccf_folder)
    
    print('##################################')
    print(conf.batch_size)
    
    weights = []
    for i in range(4):
        weights += [sum(class_num)//class_num[i] for j in range(imgs_num[i])]
    print(len(ds))
    print(len(weights))
    assert len(ds) == len(weights)
    weights = torch.FloatTensor(weights)
    
    train_sampler = WeightedRandomSampler(weights, len(ds), replacement=True)
    loader = DataLoader(
        ds, 
        batch_size=conf.batch_size, 
        sampler = train_sampler, 
        pin_memory=conf.pin_memory, 
        num_workers=conf.num_workers
        )

    if isinstance(class_num, list):
        class_num = sum(class_num)
    return loader, class_num 
    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    # max_idx = int(header.label[0])
    max_idx = 10
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)

# class train_dataset(Dataset):
#     def __init__(self, imgs_bcolz, label_bcolz, h_flip=True):
#         self.imgs = bcolz.carray(rootdir = imgs_bcolz)
#         self.labels = bcolz.carray(rootdir = label_bcolz)
#         self.h_flip = h_flip
#         self.length = len(self.imgs) - 1
#         if h_flip:
#             self.transform = trans.Compose([
#                 trans.ToPILImage(),
#                 trans.RandomHorizontalFlip(),
#                 trans.ToTensor(),
#                 trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])
#         self.class_num = self.labels[-1] + 1
        
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, index):
#         img = torch.tensor(self.imgs[index+1], dtype=torch.float)
#         label = torch.tensor(self.labels[index+1], dtype=torch.long)
#         if self.h_flip:
#             img = de_preprocess(img)
#             img = self.transform(img)
#         return img, label


def get_pair_ranking_dataset(conf, dataset_folder):
    # for pair sampling
    ds = OrderedDict()
    class_num = []
    imgs_num = []

    for path in dataset_folder.iterdir():
        if path.is_file():
            continue
        else:
            ds_tmp, class_num_tmp = get_train_dataset(path)
            race_name = path.name
            ds[race_name] = ds_tmp
            class_num.append(class_num_tmp)
            imgs_num.append(len(ds_tmp))

    label_to_race = {} 
    for j, ds_key in enumerate(ds):
        for i, (url, label) in enumerate(ds[ds_key].imgs):
            if j > 0:
                new_label = label + sum(class_num[:j])
                ds[ds_key].imgs[i] = (url, new_label)
            else:
                new_label = label
            label_to_race[new_label] = ds_key

    sub_ds = [z[1] for z in ds.items()]
    ds = ExtendConcatDataset(sub_ds)
    
    # for pair sampling
    label_to_pos_indexs = defaultdict(set) # a set containing all positive img index
    race_to_index = defaultdict(set)
    len_of_concat = len(ds)
    for idx in tqdm(range(len_of_concat)):
        img, label = ds.get_img_info(idx)
    # for idx, (img_path, label) in enumerate(ds.img):
        label_to_pos_indexs[label].add(idx)
        race = label_to_race[label]
        race_to_index[race].add(idx)

    # checkpoint
    rand_img = random.randint(0, len_of_concat - 1)
    img, label = ds.get_img_info(rand_img)
    race = label_to_race[label]
    print('img: {}, race: {}, pos num: {}'.format(img, race, len(label_to_pos_indexs[label])))

    map_dict = {
        'label2posidx': label_to_pos_indexs,
        'race2idx': race_to_index,
        'label2race': label_to_race
    }
    
    return PairPoolDataset(conf, ds, map_dict), class_num, imgs_num


class PairPoolDataset(Dataset):
    def __init__(self, conf, sequential_dataset, map_dict):
        self.dataset = sequential_dataset

        self.label_to_pos_idx = map_dict['label2posidx']
        self.label_to_race = map_dict['label2race']
        self.race_to_idx = map_dict['race2idx']
        self.pool_set = set(range(len(self.dataset)))

        self.pos_num = conf.rank_pos_num
        self.neg_num = conf.rank_neg_num
        self.race_weight = conf.race_neg_sampling_weight
        self.sampling_weight = None
        if self.race_weight is not None:
            self.sampling_weight = [[], []]
            for k, v in self.race_weight.items():
                self.sampling_weight[0].append(k)
                self.sampling_weight[1].append(v)
            
            # auto normalization
            weight = np.array(self.sampling_weight[1])
            weight = weight / np.sum(weight)
            self.sampling_weight[1] = weight.tolist()
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        # print(sample.shape, target)

        label = target
        pos_idx_set = self.label_to_pos_idx[label]
        
        if self.race_weight is None:
            pos_pool = pos_idx_set - set([index])
            neg_pool = self.pool_set - pos_idx_set

            pos_index = random.choices(list(pos_pool), k=self.pos_num)
            neg_index = random.choices(list(neg_pool), k=self.neg_num)
        else:
            pos_pool = pos_idx_set - set([index])
            neg_pool_num = np.random.multinomial(self.neg_num, self.sampling_weight[1]).tolist()

            pos_index = random.choices(list(pos_pool), k=self.pos_num)
            neg_index = []
            for k, num in zip(self.sampling_weight[0], neg_pool_num):
                neg_pool = self.race_to_idx[k] - pos_idx_set
                neg_index.extend(random.choices(list(neg_pool), k=num))
        
        pair_sample = [sample]
        pair_target = [target]
        extend_index = pos_index + neg_index
        for idx in extend_index:
            sample, target = self.dataset[idx]
            pair_sample.append(sample)
            pair_target.append(target)

        pair_sample = torch.stack(pair_sample, dim=0)
        pair_target = torch.tensor(pair_target, dtype=torch.long)

        return pair_sample, pair_target


class ExtendConcatDataset(ConcatDataset):
    def get_img_info(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].imgs[sample_idx]