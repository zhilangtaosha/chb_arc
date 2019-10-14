from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from verifacation import evaluate
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
from models_101 import resnet18, resnet34, resnet50, resnet101,ArcMarginModel

def judge_race(conf,label):
    for i in range(3):
        if label < sum(conf.race_index[:(i+1)]):
            return i
        else :
            return 3

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        self.lr=conf.lr
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
        ###############################  ir_se50  ########################################
            if conf.struct =='ir_se_50':
                self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            
                print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        ###############################  resnet101  ######################################
            if conf.struct =='ir_se_101':
                self.model = resnet101().to(conf.device)
                print('resnet101 model generated')
            
        
        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)        

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            
        ###############################  ir_se50  ########################################
            if conf.struct =='ir_se_50':
                self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
                self.head_race = Arcface(embedding_size=conf.embedding_size, classnum=4).to(conf.device)
        
        ###############################  resnet101  ######################################
            if conf.struct =='ir_se_101':
                self.head = ArcMarginModel(embedding_size=conf.embedding_size,classnum=self.class_num).to(conf.device)
                self.head_race = ArcMarginModel(embedding_size=conf.embedding_size,classnum=self.class_num).to(conf.device)
            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
            
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.kernel] + [self.head_race.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.kernel] + [self.head_race.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            print(self.optimizer)
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')    
            print('len of loader:',len(self.loader)) 
            self.board_loss_every = len(self.loader)//min(len(self.loader),100)
            self.evaluate_every = len(self.loader)//1
            self.save_every = len(self.loader)//1
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(conf.val_folder)
        else:
            #self.threshold = conf.threshold
            pass
    



    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.head_race.state_dict(), save_path /
                ('head__race{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
    
    def load_state(self, model, head=None,head_race=None,optimizer=None):
           
        self.model.load_state_dict(torch.load(model),strict=False)
        if head is not None:
            self.head.load_state_dict(torch.load(head))
        if head_race is not None:
            self.head_race.load_state_dict(torch.load(head_race))
        if optimizer is not None:
            self.optimizer.load_state_dict(torch.load(optimizer))

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor,tpr_val):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        
        self.writer.add_scalar('{}_tpr@0.001'.format(db_name), tpr_val, self.step)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
        
    def evaluate(self, conf, carray, issame, nrof_folds = 5, tta = False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        try:
            tpr_val = tpr[np.less(fpr,0.0012)&np.greater(fpr,0.0008)][0]
            
        except:
            tpr_val = 0
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor,tpr_val
    
    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)          
          
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            #Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses    

    def train(self, conf, epochs):
        self.model = self.model.to(conf.device)
        self.head = self.head.to(conf.device)
        self.head_race = self.head_race.to(conf.device)
        self.model.train()
        self.head.train()
        self.head_race.train()
        running_loss = 0.     
        for e in range(epochs):
                
        
            print('epoch {} started'.format(e))
            
            if e == 8:#5 #train hear_race
                #self.init_lr()
                conf.loss0 = False
                conf.loss1 = True
                conf.loss2 = True
                conf.model = False
                conf.head = False
                conf.head_race = True
                print(conf)
            if e == 16:#10:
                #self.init_lr()
                self.schedule_lr()
                conf.loss0 = True
                conf.loss1 = True
                conf.loss2 = True
                conf.model = True
                conf.head = True
                conf.head_race = True
                print(conf)
            if e == 28:#22
                self.schedule_lr()
            if e == 32:
                self.schedule_lr()      
            if e == 35:
                self.schedule_lr()      
            
            requires_grad(self.head,conf.head)
            requires_grad(self.head_race,conf.head_race)
            requires_grad(self.model,conf.model)                            
            for imgs, labels  in tqdm(iter(self.loader)):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                labels_race = torch.zeros_like(labels)
                
                race0_index = labels.lt(sum(conf.race_num[:1]))
                race1_index = labels.lt(sum(conf.race_num[:2])) & labels.ge(sum(conf.race_num[:1]))
                race2_index = labels.lt(sum(conf.race_num[:3])) & labels.ge(sum(conf.race_num[:2]))
                race3_index = labels.ge(sum(conf.race_num[:3]))
                labels_race[race0_index]=0
                labels_race[race1_index] = 1
                labels_race[race2_index] = 2
                labels_race[race3_index] = 3

                
                
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas ,w = self.head(embeddings, labels)
                thetas_race ,w_race = self.head_race(embeddings, labels_race)
                loss = 0
                loss0 = conf.ce_loss(thetas, labels) 
                loss1 = conf.ce_loss(thetas_race, labels_race)
                loss2 = torch.mm(w_race.t(),w).to(conf.device)
                
                target =  torch.zeros_like(loss2).to(conf.device)
                
                target[0][:sum(conf.race_num[:1])] = 1
                target[1][sum(conf.race_num[:1]):sum(conf.race_num[:2])] = 1
                target[2][sum(conf.race_num[:2]):sum(conf.race_num[:3])] = 1
                target[3][sum(conf.race_num[:3]):] = 1
                
                weight = torch.zeros_like(loss2).to(conf.device)
                for i in range(4):
                    weight[i,:] = sum(conf.race_num)/conf.race_num[i] 
                #loss2 = torch.nn.functional.mse_loss(loss2 , target)
                
                loss2 = F.binary_cross_entropy(torch.sigmoid(loss2),target,weight)
                if conf.loss0 ==True:
                    loss += 2*loss0
                if conf.loss1 ==True:
                    loss += loss1
                if conf.loss2 ==True:
                    loss += loss2
                #loss = loss0 + loss1 + loss2
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy=None
                    accuracy, best_threshold, roc_curve_tensor ,tpr_val= self.evaluate(conf, self.agedb_30, self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor,tpr_val)
                    accuracy, best_threshold, roc_curve_tensor,tpr_val = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor,tpr_val)
                    accuracy, best_threshold, roc_curve_tensor,tpr_val = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor,tpr_val)
                    self.model.train()
                    
                if self.step % self.save_every == 0 and self.step != 0:
                    
                    self.save_state(conf, accuracy)
                self.step += 1
                
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
        
    def init_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] = self.lr
        print(self.optimizer)
        
        
    def schedule_lr_add(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] *= 10
        print(self.optimizer)
        
        
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               