import argparse
from train_tfvaegan_inductive import run


# generalized ZSL
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset AWA2 --few_train False --num_shots 0 --generalized True > awa2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset SUN --few_train False --num_shots 0 --generalized True > sun.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset CUB --few_train False --num_shots 0 --generalized True > cub.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset aPY --few_train False --num_shots 0 --generalized True > apy.log 2>&1 &


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


# few shot + class
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train False --num_shots 1 --generalized True --image_embedding res101_reg --class_embedding att_GRU_biased > flo0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train False --num_shots 5 --generalized True --image_embedding res101_reg --class_embedding att_GRU_biased > flo1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train False --num_shots 10 --generalized True --image_embedding res101_reg --class_embedding att_GRU_biased > flo2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train False --num_shots 20 --generalized True --image_embedding res101_reg --class_embedding att_GRU_biased > flo3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset FLO --few_train True --num_shots 1 --generalized True --image_embedding res101_naive --class_embedding att_GRU_biased > flo0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset FLO --few_train True --num_shots 5 --generalized True --image_embedding res101_naive --class_embedding att_GRU_biased > flo1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset FLO --few_train True --num_shots 10 --generalized True --image_embedding res101_naive --class_embedding att_GRU_biased > flo2.log 2>&1 &
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

        self.dataroot = '../data'; self.syn_num = 100
        self.preprocessing = True; self.standardization = False; self.workers = 8; self.batch_size = 64; self.attSize = 1024
        self.nz = 312; self.ngh = 4096; self.ndh = 1024; self.nepoch = 2000; self.critic_iter = 5; self.lambda1 = 10
        self.lambda2 = 10; self.lr = 0.001; self.feed_lr = 0.0001; self.dec_lr = 0.0001; self.classifier_lr = 0.001
        self.beta1 = 0.5; self.cuda = True; self.encoded_noise = False; self.manualSeed = 0; self.nclass_all = 200
        self.validation = False; self.encoder_layer_sizes = [8192, 4096]; self.decoder_layer_sizes = [4096, 8192]
        self.gammaD = 1000; self.gammaG = 1000; self.gammaG_D2 = 1000; self.gammaD2 = 1000; self.latent_size = 312
        self.latent_size = 312; self.conditional = True; self.a1 = 1.0; self.a2 = 1.0; self.recons_weight = 1.0
        self.recons_weight = 1.0; self.feedback_loop = 2; self.freeze_dec = False; self.val_interval = 10
        self.save_interval = 50; self.continue_from = 0; self.debug = False; self.additional = ""; self.encoder_use_y = False
        self.train_deterministic = False; self.z_disentangle = False; self.zd_beta = 1.0; self.zd_beta = False
        self.zd_beta_annealing = False; self.zy_disentangle = False; self.zy_lambda = 0.01; self.yz_disentangle = False
        self.yz_lambda = 0.01; self.yz_celoss = False; self.yx_disentangle = False; self.yx_lambda = 0.01; self.zx_disentangle = False
        self.zx_lambda = 0.01; self.contrastive_loss = False; self.temperature = 1.0; self.contra_lambda = 1.0; self.contra_v = 3
        self.K = 30; self.siamese_loss = False; self.siamese_lambda = 1.0; self.siamese_use_softmax = False
        self.siamese_distance = "l1"; self.pca_attribute = 0; self.survae = False; self.m_lambda = 100; self.add_noise = 0.0
        self.recon = "bce"; self.attdec_use_z = False; self.attdec_use_mse = False; self.z_loss = False; self.z_loss_lambda = 1.0
        self.p_loss = False; self.p_loss_lambda = 1.0; self.eval = False; self.two_stage = False; self.clf_epoch = 5
        self.concat_hy = 1; self.sanity = False; self.baseline = False; self.cf_eval = ""; self.report_softmax = False
        self.report_knn = False; self.report_knn = False; self.report_gzsl = False; self.report_zsl = False
        self.test_deterministic = False; self.use_mask = None; self.use_train = 1; self.binary = False
        self.siamese = False; self.load_best_acc = False; self.save_auroc = False; self.save_auroc_cf = False
        self.analyze_auroc = False; self.analyze_auroc_cf = False; self.analyze_auroc_expname = None
        self.save_auroc_mask = False; self.use_tde = False; self.tde_alpha = 0.5; self.log_two_stage = False
        self.u_num = 400; self.u_lr = 0.5; self.u_beta = 0.5; self.u_epoch = 2; self.u_batch_size = 400
        self.adjust_s = False; self.s_lr = 0.001; self.s_beta = 0.5; self.s_epoch = 2; self.s_batch_size = 400
        self.resSize = 2048

        if self.dataset in ["AWA1","AWA2"]:
            self.gammaD = 10; self.gammaG = 10; self.encoded_noise = True; self.manualSeed = 9182; self.preprocessing = True
            self.cuda = True; self.nepoch = 200; self.syn_num = 1800
            self.ngh = 4096; self.ndh = 4096; self.lambda1 = 10; self.critic_iter = 5; self.nclass_all = 50; self.dataroot = "../data"
            self.batch_size = 64; self.nz = 85; self.latent_size = 85; self.attSize = 85; self.resSize = 2048; self.encoder_use_y = True
            self.lr = 0.00001; self.classifier_lr = 0.001; self.recons_weight = 0.1; self.freeze_dec = True; self.save_interval = 40
            self.feed_lr = 0.0001; self.dec_lr = 0.0001; self.feedback_loop = 2; self.a1 = 0.01; self.a2 = 0.01; self.val_interval = 5
            self.clf_epoch = 5
        elif self.dataset == "CUB":
            self.gammaD = 10; self.gammaG = 10; self.manualSeed = 3483; self.encoded_noise = True; self.preprocessing = True
            self.cuda = True; self.nepoch = 300; self.ngh = 4096
            self.ndh = 4096; self.lr = 0.0001; self.classifier_lr = 0.001; self.lambda1 = 15; self.critic_iter = 5; self.nclass_all = 200
            self.batch_size = 64; self.nz = 312; self.latent_size = 312; self.attSize = 312; self.resSize = 2048; self.syn_num = 300
            self.recons_weight = 0.01; self.a1 = 1; self.a2 = 1; self.feed_lr = 0.00001; self.dec_lr = 0.0001; self.feedback_loop = 2
            self.val_interval = 5; self.clf_epoch = 5; self.encoder_use_y = True; self.baseline = True
        elif self.dataset == "aPY":
            self.gammaD = 10; self.gammaG = 10; self.encoded_noise = True; self.manualSeed = 9182; self.preprocessing = True
            self.cuda = True;  self.nepoch = 300; self.syn_num = 1800
            self.ngh = 4096; self.ndh = 4096; self.lambda1 = 10; self.critic_iter = 5; self.nclass_all = 32; self.batch_size = 64
            self.nz = 64; self.latent_size = 64; self.attSize = 64; self.resSize = 2048; self.encoder_use_y = True; self.lr = 0.00001
            self.classifier_lr = 0.001; self.recons_weight = 0.1; self.save_interval = 50; self.feed_lr = 0.0001; self.dec_lr = 0.0005
            self.feedback_loop = 2; self.a1 = 0.01; self.a2 = 0.01; self.val_interval = 5; self.clf_epoch = 2
            self.additional = "lre-5_a1e-2_a2e-2_recon0.1_dec5e-4"
        elif self.dataset == "SUN":
            self.gammaD = 1; self.gammaG = 1; self.manualSeed = 4115; self.encoded_noise = True; self.preprocessing = True
            self.cuda = True; self.nepoch = 400
            self.ngh = 4096; self.lambda1 = 10; self.critic_iter = 5; self.batch_size = 64;  self.latent_size = 102
            self.lr = 0.001; self.classifier_lr = 0.0005; self.syn_num = 400; self.nclass_all = 717; self.recons_weight = 0.01
            self.a1 = 0.1; self.a2 = 0.01; self.feedback_loop = 2; self.feed_lr = 0.0001; self.encoder_use_y = True
            self.save_interval = 50; self.clf_epoch = 5; self.val_interval = 5; self.resSize = 2048; self.attSize = 102
            self.nz = 102
            if self.image_embedding == "res101_reg":
                self.lr = 0.0001
        elif self.dataset == "FLO":
            self.gammaD = 10; self.gammaG = 10; self.encoded_noise = True; self.manualSeed = 9182; self.preprocessing = True
            self.cuda = True; self.nepoch = 200; self.syn_num = 1800
            self.ngh = 4096; self.ndh = 4096; self.lambda1 = 10; self.critic_iter = 5; self.nclass_all = 102; self.dataroot = "../data"
            self.batch_size = 64; self.nz = 1024; self.latent_size = 1024; self.attSize = 1024; self.resSize = 2048; self.encoder_use_y = True
            self.lr = 0.00001; self.classifier_lr = 0.001; self.recons_weight = 0.1; self.freeze_dec = True; self.save_interval = 40
            self.feed_lr = 0.0001; self.dec_lr = 0.0001; self.feedback_loop = 2; self.a1 = 0.01; self.a2 = 0.01; self.val_interval = 5
            self.clf_epoch = 5


opt = myArgs(args)
opt.lambda2 = opt.lambda1
opt.encoder_layer_sizes[0] = opt.resSize
opt.decoder_layer_sizes[-1] = opt.resSize

print(opt.lr, opt.classifier_lr, opt.lambda1, opt.lambda2, opt.syn_num, opt.recons_weight, opt.a1, opt.a2)

run(opt)












