from config import get_config
from Learner import face_learner
import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
import json
from pathlib import Path
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
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-4, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=128, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-p", "--pretrained", help="pretrained_model", default='./pretrained_model/model_resnet101.pth', type=str)
    parser.add_argument("-workspace", "--workspace", help="work space for model", default='work_space_ir_se_101_multihead', type=str)
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
    
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    conf.pretrained = args.pretrained
    conf.head_race_pretrained = "/home/hbchen/code/ccf/InsightFace_CCF/work_space_ir_se_101_head_race/models/head__race2019-10-10-00-08_accuracy:0.8715714285714287_step:8920_None.pth"
    conf.head_pretrained = "/home/hbchen/code/ccf/InsightFace_CCF/work_space_ir_se_101_head/models/head_2019-10-09-18-56_accuracy:0.8734285714285714_step:54226_None.pth"
    
    check_save_path(conf)
    
    write_conf(conf, conf.work_path)

    learner = face_learner(conf)
    
    #learner.load_state(model=args.pretrained , model_only=True,head=head_pretrained,head_race=head_race_pretrained)
    learner.load_state(model=args.pretrained , head=conf.head_pretrained,head_race=conf.head_race_pretrained,optimizer=None)
    
    #learner.schedule_lr()
    learner.train(conf, args.epochs,model=False,head=True,head_race=True)