import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from fclswgan import Model

# 复现命令
# generalized ZSL
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset AWA1 --few_train False --num_shots 0 --generalized True > awa1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset SUN --few_train False --num_shots 0 --generalized True > sun.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset CUB --few_train False --num_shots 0 --generalized True > cub.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized True > flo.log 2>&1 &
# conventional ZSL
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset AWA1 --few_train False --num_shots 0 --generalized False > awa1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset SUN --few_train False --num_shots 0 --generalized False > sun.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset CUB --few_train False --num_shots 0 --generalized False > cub.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized False > flo.log 2>&1 &

# naive feature
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset AWA2 --few_train False --num_shots 0 --generalized True --image_embedding res101_naive > awa2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset SUN --few_train False --num_shots 0 --generalized True --image_embedding res101_naive > sun.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset CUB --few_train False --num_shots 0 --generalized True --image_embedding res101_naive > cub.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized True --image_embedding res101_naive > flo.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset aPY --few_train False --num_shots 0 --generalized True --image_embedding res101_naive > apy.log 2>&1 &

# finetue feature
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset AWA2 --few_train False --num_shots 0 --generalized True --image_embedding res101_finetune > awa2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset SUN --few_train False --num_shots 0 --generalized True --image_embedding res101_finetune > sun.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset CUB --few_train False --num_shots 0 --generalized True --image_embedding res101_finetune > cub.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized True --image_embedding res101_finetune > flo.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset aPY --few_train False --num_shots 0 --generalized True --image_embedding res101_finetune > apy.log 2>&1 &

# reg feature
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized True --image_embedding res101_reg > flo.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset CUB --few_train False --num_shots 0 --generalized True --image_embedding res101_reg > cub.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset SUN --few_train False --num_shots 0 --generalized True --image_embedding res101_reg > sun.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset AWA2 --few_train False --num_shots 0 --generalized True --image_embedding res101_reg > awa2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset aPY --few_train False --num_shots 0 --generalized True --image_embedding res101_reg > apy.log 2>&1 &

# reg feature + att
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized True --image_embedding res101_reg --class_embedding att > flo0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized True --image_embedding res101_reg --class_embedding att_naive > flo1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized True --image_embedding res101_reg --class_embedding att_GRU > flo2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized True --image_embedding res101_reg --class_embedding att_GRU_biased > flo3.log 2>&1 &

# few shot
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train False --num_shots 1 --generalized True --image_embedding res101_reg > flo0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train False --num_shots 5 --generalized True --image_embedding res101_reg > flo1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset FLO --few_train False --num_shots 10 --generalized True --image_embedding res101_reg > flo2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset FLO --few_train False --num_shots 20 --generalized True --image_embedding res101_reg > flo3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train True --num_shots 1 --generalized True --image_embedding res101_naive > flo0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train True --num_shots 5 --generalized True --image_embedding res101_naive > flo1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset FLO --few_train True --num_shots 10 --generalized True --image_embedding res101_naive > flo2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset FLO --few_train True --num_shots 20 --generalized True --image_embedding res101_naive > flo3.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset CUB --few_train False --num_shots 1 --generalized True --image_embedding res101_reg > cub0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset CUB --few_train False --num_shots 5 --generalized True --image_embedding res101_reg > cub1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset CUB --few_train False --num_shots 10 --generalized True --image_embedding res101_reg > cub2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset CUB --few_train False --num_shots 20 --generalized True --image_embedding res101_reg > cub3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset CUB --few_train True --num_shots 1 --generalized True --image_embedding res101_naive > cub0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset CUB --few_train True --num_shots 5 --generalized True --image_embedding res101_naive > cub1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset CUB --few_train True --num_shots 10 --generalized True --image_embedding res101_naive > cub2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset CUB --few_train True --num_shots 20 --generalized True --image_embedding res101_naive > cub3.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset SUN --few_train False --num_shots 1 --generalized True --image_embedding res101_reg > sun0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset SUN --few_train False --num_shots 5 --generalized True --image_embedding res101_reg > sun1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset SUN --few_train False --num_shots 10 --generalized True --image_embedding res101_reg > sun2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset SUN --few_train True --num_shots 1 --generalized True --image_embedding res101 > sun0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset SUN --few_train True --num_shots 5 --generalized True --image_embedding res101 > sun1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset SUN --few_train True --num_shots 10 --generalized True --image_embedding res101 > sun2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset AWA2 --few_train False --num_shots 1 --generalized True --image_embedding res101_naive > awa20.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset AWA2 --few_train False --num_shots 5 --generalized True --image_embedding res101_naive > awa21.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset AWA2 --few_train False --num_shots 10 --generalized True --image_embedding res101_naive > awa22.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset AWA2 --few_train False --num_shots 20 --generalized True --image_embedding res101_naive > awa23.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset AWA2 --few_train True --num_shots 1 --generalized True --image_embedding res101_naive > awa20.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset AWA2 --few_train True --num_shots 5 --generalized True --image_embedding res101_naive > awa21.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset AWA2 --few_train True --num_shots 10 --generalized True --image_embedding res101_naive > awa22.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset AWA2 --few_train True --num_shots 20 --generalized True --image_embedding res101_naive > awa23.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset AWA1 --few_train False --num_shots 1 --generalized True --image_embedding res101 > awa10.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset AWA1 --few_train False --num_shots 5 --generalized True --image_embedding res101 > awa11.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset AWA1 --few_train False --num_shots 10 --generalized True --image_embedding res101 > awa12.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset AWA1 --few_train False --num_shots 20 --generalized True --image_embedding res101 > awa13.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset AWA1 --few_train True --num_shots 1 --generalized True --image_embedding res101 > awa10.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset AWA1 --few_train True --num_shots 5 --generalized True --image_embedding res101 > awa11.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset AWA1 --few_train True --num_shots 10 --generalized True --image_embedding res101 > awa12.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset AWA1 --few_train True --num_shots 20 --generalized True --image_embedding res101 > awa13.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized False --image_embedding res101_reg --class_embedding att > flo0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized False --image_embedding res101_reg --class_embedding att_naive > flo1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized False --image_embedding res101_reg --class_embedding att_GRU > flo2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized False --image_embedding res101_reg --class_embedding att_GRU_biased > flo3.log 2>&1 &






# few shot + class
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train False --num_shots 1 --generalized True --image_embedding res101_reg --class_embedding att_GRU_biased > flo0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train False --num_shots 5 --generalized True --image_embedding res101_reg --class_embedding att_GRU_biased > flo1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train False --num_shots 10 --generalized True --image_embedding res101_reg --class_embedding att_GRU_biased > flo2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train False --num_shots 20 --generalized True --image_embedding res101_reg --class_embedding att_GRU_biased > flo3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train True --num_shots 1 --generalized True --image_embedding res101_naive --class_embedding att_GRU_biased > flo0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train True --num_shots 5 --generalized True --image_embedding res101_naive --class_embedding att_GRU_biased > flo1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train True --num_shots 10 --generalized True --image_embedding res101_naive --class_embedding att_GRU_biased > flo2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train True --num_shots 20 --generalized True --image_embedding res101_naive --class_embedding att_GRU_biased > flo3.log 2>&1 &




def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--few_train', default = False, type = str2bool, help='use few train samples')
parser.add_argument('--num_shots', type=int, default=5, help='the number of shots, if few_train, then num_shots is for train classes, else for test classes')
parser.add_argument('--generalized', default=False, type = str2bool, help='enable generalized zero-shot learning')
parser.add_argument('--image_embedding', default='res101', help='res101')
parser.add_argument('--class_embedding', default='att', help='att')
opt_par = parser.parse_args()
print(opt_par)

print("The current working directory is")
print(os.getcwd())
folder = str(Path(os.getcwd()))
project_directory = Path(os.getcwd()).parent
print('Project Directory:')
print(project_directory)
data_path = str(project_directory) + '/data/'
print('Data Path')
print(data_path)

opt = {
    'dataset':opt_par.dataset, 'few_train':opt_par.few_train, 'num_shots':opt_par.num_shots,
    'image_embedding':opt_par.image_embedding,
    'generalized':opt_par.generalized, 'dataroot': data_path, 'class_embedding':opt_par.class_embedding,
    'matdataset': True, 'syn_num':100,
    'preprocessing':False, 'standardization':False, 'validation':False, 'workers':4, 'batch_size':64,
    'resSize':2048, 'attSize':1024, 'nz':312, 'ngh':4096, 'ndh':1024, 'nepoch':2000, 'critic_iter':5,
    'lambda1':10, 'cls_weight':1, 'lr':0.0001, 'classifier_lr':0.001, 'beta1':0.5, 'cuda':True, 'ngpu':1,
    'print_every':1, 'start_epoch':0, 'manualSeed':1, 'nclass_all':200}

if opt['generalized'] == True:
    if opt['dataset'] in ['AWA1','AWA2']:
        opt['manualSeed'] = 9182; opt['cls_weight'] = 0.01; opt['lr'] = 0.00001; opt['nepoch'] = 300
        opt['syn_num'] = 4000; opt['ngh'] = 4096; opt['ndh'] = 4096; opt['lambda1'] = 10; opt['critic_iter'] = 5
        opt['nclass_all'] = 50; opt['batch_size'] = 64; opt['nz'] = 85; opt['attSize'] = 85; opt['resSize'] = 2048
    elif opt['dataset'] == 'CUB':
        opt['manualSeed'] = 3483; opt['cls_weight'] = 0.01; opt['lr'] = 0.0001; opt['nepoch'] = 600
        opt['syn_num'] = 1500; opt['ngh'] = 4096; opt['ndh'] = 4096; opt['lambda1'] = 10; opt['critic_iter'] = 5
        opt['nclass_all'] = 200; opt['dataset'] = 'CUB'; opt['batch_size'] = 64
        opt['resSize'] = 2048; opt['classifier_lr'] = 0.001; opt['attSize'] = 312; opt['nz'] = 312
    elif opt['dataset'] == 'FLO':
        opt['manualSeed'] = 806; opt['cls_weight'] = 0.1; opt['lr'] = 0.0001; opt['nepoch'] = 800
        opt['syn_num'] = 1200; opt['ngh'] = 4096; opt['ndh'] = 4096; opt['lambda1'] = 10; opt['critic_iter'] = 5
        opt['nclass_all'] = 102; opt['dataset'] = 'FLO'; opt['batch_size'] = 64; opt['nz'] = 1024
        opt['attSize'] = 1024; opt['resSize'] = 2048; opt['classifier_lr'] = 0.001
    elif opt['dataset'] == 'SUN':
        opt['manualSeed'] = 4115; opt['cls_weight'] = 0.01; opt['lr'] = 0.0002; opt['nepoch'] = 400
        opt['syn_num'] = 400; opt['ngh'] = 4096; opt['ndh'] = 4096; opt['lambda1'] = 10; opt['critic_iter'] = 5
        opt['nclass_all'] = 717;  opt['dataset'] = 'SUN'; opt['batch_size'] = 64; opt['nz'] = 102;
        opt['attSize'] = 102; opt['resSize'] = 2048; opt['classifier_lr'] = 0.0001
    elif opt['dataset'] == 'aPY':
        opt['manualSeed'] = 4115; opt['cls_weight'] = 0.01; opt['lr'] = 0.0002; opt['nepoch'] = 400
        opt['syn_num'] = 2400; opt['ngh'] = 4096; opt['ndh'] = 4096; opt['lambda1'] = 10; opt['critic_iter'] = 5
        opt['nclass_all'] = 32;  opt['dataset'] = 'aPY'; opt['batch_size'] = 64; opt['nz'] = 64;
        opt['attSize'] = 64; opt['resSize'] = 2048; opt['classifier_lr'] = 0.001
else:
    if opt['dataset'] in ['AWA1','AWA2']:
        opt['manualSeed'] = 9182; opt['cls_weight'] = 0.01; opt['lr'] = 0.00001; opt['nepoch'] = 301
        opt['syn_num'] = 4000; opt['ngh'] = 4096; opt['ndh'] = 4096; opt['lambda1'] = 10; opt['critic_iter'] = 5
        opt['nclass_all'] = 50; opt['dataset'] = 'AWA1'; opt['batch_size'] = 64; opt['nz'] = 85
        opt['attSize'] = 85; opt['resSize'] = 2048
    elif opt['dataset'] == 'CUB':
        opt['manualSeed'] = 3483; opt['cls_weight'] = 0.01; opt['lr'] = 0.0001; opt['nepoch'] = 700
        opt['syn_num'] = 500; opt['ngh'] = 4096; opt['ndh'] = 4096; opt['lambda1'] = 10; opt['critic_iter'] = 5
        opt['nclass_all'] = 200; opt['dataset'] = 'CUB'; opt['batch_size'] = 64; opt['nz'] = 312
        opt['resSize'] = 2048; opt['classifier_lr'] = 0.001; opt['attSize'] = 312; opt['nz'] = 312
    elif opt['dataset'] == 'FLO':
        opt['manualSeed'] = 806; opt['cls_weight'] = 0.1; opt['lr'] = 0.0001; opt['nepoch'] = 970
        opt['syn_num'] = 300; opt['ngh'] = 4096; opt['ndh'] = 4096; opt['lambda1'] = 10; opt['critic_iter'] = 5
        opt['nclass_all'] = 102; opt['dataset'] = 'FLO'; opt['batch_size'] = 64; opt['nz'] = 1024
        opt['attSize'] = 1024; opt['resSize'] = 2048; opt['classifier_lr'] = 0.001
    elif opt['dataset'] == 'SUN':
        opt['manualSeed'] = 4115; opt['cls_weight'] = 0.01; opt['lr'] = 0.0002; opt['nepoch'] = 540
        opt['syn_num'] = 400; opt['ngh'] = 4096; opt['ndh'] = 4096; opt['lambda1'] = 10; opt['critic_iter'] = 5
        opt['nclass_all'] = 717;  opt['dataset'] = 'SUN'; opt['batch_size'] = 64; opt['nz'] = 102;
        opt['attSize'] = 102; opt['resSize'] = 2048; opt['classifier_lr'] = 0.001; opt['beta1'] = 0.5
    elif opt['dataset'] == 'aPY':
        opt['manualSeed'] = 4115; opt['cls_weight'] = 0.01; opt['lr'] = 0.0002; opt['nepoch'] = 400
        opt['syn_num'] = 2400; opt['ngh'] = 4096; opt['ndh'] = 4096; opt['lambda1'] = 10; opt['critic_iter'] = 5
        opt['nclass_all'] = 32;  opt['dataset'] = 'aPY'; opt['batch_size'] = 64; opt['nz'] = 64;
        opt['attSize'] = 64; opt['resSize'] = 2048; opt['classifier_lr'] = 0.001
print("**********")
print(opt['cls_weight'], opt['lr'], opt['syn_num'], opt['lambda1'], opt['batch_size'], opt['classifier_lr'])
print("**********")
cudnn.benchmark = True
if torch.cuda.is_available() and not opt['cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

my_model = Model(opt)
my_model.train_fclswgan()

