import argparse
from lisgan import run

# generalized ZSL
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset AWA1 --few_train False --num_shots 0 --generalized True > awa1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset SUN --few_train False --num_shots 0 --generalized True > sun.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset CUB --few_train False --num_shots 0 --generalized True > cub.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset FLO --few_train False --num_shots 0 --generalized True > flo.log 2>&1 &
# ZSL
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
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset AWA2 --few_train False --num_shots 0 --generalized True --image_embedding res101_reg > awa2.log 2>&1 &
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
args = parser.parse_args()

class myArgs():
    def __init__(self, args):
        self.dataset = args.dataset
        self.few_train = args.few_train
        self.num_shots = args.num_shots
        self.generalized = args.generalized
        self.image_embedding = args.image_embedding
        self.class_embedding = args.class_embedding
        self.dataroot = '../data'; self.matdataset = True
        self.syn_num = 100; self.preprocessing = False; self.standardization = False; self.validation = False; self.workers = 2
        self.batch_size = 64; self.resSize = 2048; self.attSize = 1024; self.nz = 312; self.ngh = 4096
        self.ndh = 1024; self.nepoch = 2000; self.critic_iter = 5; self.lambda1 = 10; self.cls_weight = 1
        self.lr = 0.0001; self.classifier_lr = 0.001; self.beta1 = 0.5; self.cuda = False; self.ngpu = 1
        self.pretrain_classifier = ''; self.netG = ''; self.netD = ''; self.netG_name = ''; self.netD_name = ''
        self.outf = './checkpoint/'; self.outname = ''; self.save_every = 100; self.print_every = 1; self.val_every = 10
        self.start_epoch = 0; self.manualSeed = 0; self.nclass_all = 200; self.ratio = 0.2; self.proto_param1 = 0.01
        self.proto_param2 = 0.01; self.loss_syn_num = 20; self.n_clusters = 3

        if self.generalized:
            if self.dataset == 'CUB':
                self.proto_param1 = 1e-2; self.proto_param2 = 0.001; self.ratio = 0.2; self.manualSeed = 3483;
                self.val_every = 1; self.cls_weight = 0.01; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.0001; self.classifier_lr = 0.001; self. lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 200; self.batch_size = 64; self.nz = 312; self.attSize = 312
                self.resSize = 2048; self.syn_num = 300;self.outname = 'cub'
            elif self.dataset == 'FLO':
                self.proto_param1 = 1e-1; self.proto_param2 = 3e-2; self.ratio = 0.4; self.manualSeed = 806;
                self.val_every = 1; self.cls_weight = 0.1; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.0001; self.classifier_lr = 0.001; self. lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 102; self.batch_size = 64; self.nz = 1024; self.attSize = 1024
                self.resSize = 2048; self.syn_num = 1200;self.outname = 'flower'
            elif self.dataset == 'SUN':
                self.proto_param1 = 3e-1; self.proto_param2 = 3e-5; self.ratio = 0.1; self.manualSeed = 4115;
                self.val_every = 1; self.cls_weight = 0.01; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.0002; self.classifier_lr = 0.001; self. lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 717; self.batch_size = 64; self.nz = 102; self.attSize = 102
                self.resSize = 2048; self.syn_num = 400; self.outname = 'sun'
            elif self.dataset == 'AWA1':
                self.proto_param1 = 1e-3; self.proto_param2 = 3e-5; self.ratio = 0.1; self.manualSeed = 9182;
                self.val_every = 1; self.cls_weight = 0.01; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.00001; self.classifier_lr = 0.001; self. lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 50; self.batch_size = 64; self.nz = 85; self.attSize = 85
                self.resSize = 2048; self.syn_num = 1800; self.outname = 'awa'
            elif self.dataset == 'aPY':
                self.proto_param1 = 3e-1; self.proto_param2 = 3e-4; self.ratio = 0.2; self.manualSeed = 9182;
                self.val_every = 1; self.cls_weight = 0.01; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.00001; self.classifier_lr = 0.001; self. lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 32; self.batch_size = 64; self.nz = 64; self.attSize = 64
                self.resSize = 2048; self.syn_num = 1800; self.outname = 'apy'
            elif self.dataset == 'AWA2':
                self.proto_param1 = 1e-3; self.proto_param2 = 3e-5; self.ratio = 0.1; self.manualSeed = 9182;
                self.val_every = 1; self.cls_weight = 0.01; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.00001; self.classifier_lr = 0.001; self. lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 50; self.batch_size = 64; self.nz = 85; self.attSize = 85
                self.resSize = 2048; self.syn_num = 1800; self.outname = 'awa2'
        else:
            if self.dataset == 'CUB':
                self.proto_param1 = 1e-2; self.proto_param2 = 3e-5; self.ratio = 0.6; self.manualSeed = 3483;
                self.val_every = 1; self.cls_weight = 0.01; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.0001; self.classifier_lr = 0.001; self. lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 200; self.batch_size = 64; self.nz = 312; self.attSize = 312
                self.resSize = 2048; self.syn_num = 300;self.outname = 'cub'
            elif self.dataset == 'FLO':
                self.proto_param1 = 1e-4; self.proto_param2 = 1e-6; self.ratio = 0.6; self.manualSeed = 806;
                self.val_every = 1; self.cls_weight = 0.1; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.0001; self.classifier_lr = 0.005; self. lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 102; self.batch_size = 64; self.nz = 1024; self.attSize = 1024
                self.resSize = 2048; self.syn_num = 300;self.outname = 'flower'
            elif self.dataset == 'SUN':
                self.proto_param1 = 3e-1; self.proto_param2 = 3e-4; self.ratio = 0.5; self.manualSeed = 4115;
                self.val_every = 1; self.cls_weight = 0.01; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.0002; self.classifier_lr = 0.0005; self. lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 717; self.batch_size = 64; self.nz = 102; self.attSize = 102
                self.resSize = 2048; self.syn_num = 100; self.outname = 'sun'
            elif self.dataset == 'AWA1':
                self.proto_param1 = 3e-2; self.proto_param2 = 3e-5; self.ratio = 0.1; self.manualSeed = 9182;
                self.val_every = 1; self.cls_weight = 0.001; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.0001; self.classifier_lr = 0.001; self. lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 50; self.batch_size = 64; self.nz = 85; self.attSize = 85
                self.resSize = 2048; self.syn_num = 300; self.outname = 'awa'
            elif self.dataset == 'aPY':
                self.proto_param1 = 1; self.proto_param2 = 3e-5; self.ratio = 0.7; self.manualSeed = 9182;
                self.val_every = 1; self.cls_weight = 0.01; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.00001; self.classifier_lr = 0.001; self.lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 32; self.batch_size = 64; self.nz = 64; self.attSize = 64
                self.resSize = 2048; self.syn_num = 300; self.outname = 'apy'
            elif self.dataset == 'AWA2':
                self.proto_param1 = 3e-2; self.proto_param2 = 3e-5; self.ratio = 0.1; self.manualSeed = 9182;
                self.val_every = 1; self.cls_weight = 0.01; self.preprocessing = True; self.cuda = True;
                self.netG_name = 'MLP_G'; self.netD_name = 'MLP_CRITIC'; self.nepoch = 200
                self.ngh = 4096; self.ndh = 4096; self.lr = 0.00001; self.classifier_lr = 0.001; self.lambda1 = 10
                self.critic_iter = 5; self.nclass_all = 50; self.batch_size = 64; self.nz = 85; self.attSize = 85
                self.resSize = 2048; self.syn_num = 300; self.outname = 'awa2'


opt = myArgs(args)
print(opt.proto_param2, opt.proto_param1, opt.cls_weight, opt.lr, opt.batch_size, opt.syn_num, opt.ratio, opt.classifier_lr, opt.lambda1, opt.nz, opt.image_embedding)
run(opt)













