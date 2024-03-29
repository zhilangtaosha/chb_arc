from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(args,training = True):
    conf = edict()
    
    conf.race_num = None
    conf.data_path = Path('/data2/hbchen/ccf/training')#address of the dataset
    conf.work_path = Path(args.workspace)
    conf.model_path = conf.work_path/'models'#temp models
    conf.log_path = conf.work_path/'log_chb'
    conf.save_path = conf.work_path/'final_model'#final models
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.Resize((112,112)),
                    trans.CenterCrop((112,112)),
                    trans.ToTensor(),
                    #trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    #conf.data_mode = 'African'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.ccf_folder = conf.data_path/'ccf'
    conf.val_folder = Path('/data2/hbchen/ccf/emore_data/faces_emore')#emore val_data and train_data 
    conf.batch_size = 100
    
#   conf.batch_size = 200 # mobilefacenet
#--------------------Training Config ------------------------    
    if training:        
        
        
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.milestones = [10,15,18]
        conf.momentum = 0.9
        conf.pin_memory = False
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()    
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf