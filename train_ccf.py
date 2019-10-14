from config import get_config
from Learner import face_learner
import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
import json
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "4"

# python train.py -net mobilefacenet -b 200 -w 4

def check_save_path(conf):
    check_path = [conf.work_path, conf.model_path, conf.log_path, conf.save_path]
    for path in check_path:
        if not path.exists():
            path.mkdir()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        import torch
        from pathlib import Path
        from torchvision import transforms as trans

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bytes, torch.nn.Module, Path, torch.device, trans.Compose)):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def write_conf(conf, path: Path):
    json_file = path/'conf.json'
    with json_file.open('w') as f:
        json.dump(conf, f, cls=MyEncoder, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=45, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-4, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=50, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-p", "--pretrained", help="pretrained_model", default='/data2/hbchen/ccf/pretrained_model/model_ir_se101.pth', type=str)
    parser.add_argument("-ws", "--workspace", help="work space for model", default='/data2/hbchen/ccf/workspace_ir_se101_finetune_dist', type=str)
    parser.add_argument("-s", "--struct", help="backbone struct", default='ir_se_101', type=str)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat, ccf,African,Caucasian,Indian,Asian]",default='ccf', type=str)
    args = parser.parse_args()

    conf = get_config(args)
    conf.struct = args.struct
    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth    
    conf.epochs = args.epochs
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    conf.model = False
    conf.head = True
    conf.head_race = False
    conf.loss0=True#loss of head
    conf.loss1=False#loss of race head
    conf.loss2=False#loss of multi head
    conf.pretrained = args.pretrained
    conf.head_race_pretrained = None
    conf.head_pretrained = None
    conf.optimizer = None
    check_save_path(conf)
    
    write_conf(conf, conf.work_path)

    learner = face_learner(conf)
    
    #learner.load_state(model=args.pretrained , model_only=True,head=head_pretrained,head_race=head_race_pretrained)
    #learner.load_state(model=args.pretrained , head=conf.head_pretrained,head_race=conf.head_race_pretrained,optimizer=None)
    learner.load_state(model=args.pretrained , head=conf.head_pretrained,head_race=conf.head_race_pretrained,optimizer=conf.optimizer)
    #learner.schedule_lr()
    learner.train(conf, args.epochs)