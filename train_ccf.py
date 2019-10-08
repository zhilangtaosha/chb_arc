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
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=100, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-p", "--pretrained", help="pretrained_model", default='./pretrained_model/model_ir_se50.pth', type=str)
    parser.add_argument("-workspace", "--workspace", help="work space for model", default='work_space_ir_se', type=str)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat, ccf,African,Caucasian,Indian,Asian]",default='ccf', type=str)
    args = parser.parse_args()

    conf = get_config(args)
    
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
    check_save_path(conf)
    
    write_conf(conf, conf.work_path)

    learner = face_learner(conf)
    learner.load_state(conf, model=args.pretrained , model_only=True)
    #learner.head.load_state_dict(torch.load('./work_space/final_model/head_2019-10-07-15-40_accuracy:0.9282857142857143_step:57080_final.pth'))
    #learner.schedule_lr()
    learner.train(conf, args.epochs)