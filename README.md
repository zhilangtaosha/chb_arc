# chb_arc

resnet_ir_se50预训练模型：https://drive.google.com/file/d/1AqLb7D5eMNDSaGRIT9i1NHYvUSHm5eZv/view?usp=sharing

resnet_ir_se101预训练模型链接：https://pan.baidu.com/s/1xQ9viGRs6YiMLp9uu1aq5Q 
提取码：v9l0 

存放于
work_space/pretrainedd_model路径

REF:https://github.com/TreB1eN/InsightFace_Pytorch

https://github.com/LcenArthas/CCF-BDCI2019-Multi-person-Face-Recognition-Competition-Baseline
 
 
--------------------------------------------------------------------------------------------------------

config.py保存相关超参数，可在文件中修改

---------------------------------------------------------------------------------------------------------

训练方法：
python train_ccf.py -lr learning rate -b batch_size --pretrained address of pretrained model --workspace workspace dir --struct ir_se_101 or ir_se_50

其中workspace选项将创建一个文件夹，存放训练过程中保存的模型参数，以及tensorboarrd log，以及保存本次训练的所有超参数信息以便比较实验结果，推荐每次训练就使用一个新的workspace

struct 选项选择使用backbone，分别是ir_se_101和ir_se_50

代码中
learner.train(conf, args.epochs,model=False,head=True,head_race=True)，如若想训练中freeze住某一个网络，则将对应参数设置为false

---------------------------------------------------------------------------------------------------------
测试方法：
python test——ccf.py

--csv为模板csv   ；
--tocsv为保存的新的csv  ；
--testdir为测试集数据地址  ；
--save_mat为中途保存的测试集特征向量  ；
-ckpt 为model地址

