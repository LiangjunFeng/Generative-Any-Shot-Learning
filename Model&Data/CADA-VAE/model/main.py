# execute
# python main.py --dataset CUB --num_shots 0 --generalized True
# python main.py --dataset CUB --num_shots 0 --generalized False




from vaemodel import Model
import numpy as np
import pickle
import torch
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset')
parser.add_argument('--num_shots',type=int)
parser.add_argument('--generalized', type = str2bool)
args = parser.parse_args()


########################################
# the basic hyperparameters
########################################
hyperparameters = {
    'num_shots': 0,
    'device': 'cuda',
    'model_specifics': {'cross_reconstruction': True,
                       'name': 'CADA',
                       'distance': 'wasserstein',
                       'warmup': {'beta': {'factor': 0.25,
                                           'end_epoch': 93,
                                           'start_epoch': 0},
                                  'cross_reconstruction': {'factor': 2.37,
                                                           'end_epoch': 75,
                                                           'start_epoch': 21},
                                  'distance': {'factor': 8.13,
                                               'end_epoch': 22,
                                               'start_epoch': 6}}},

    'lr_gen_model': 0.00015,
    'generalized': True,
    'batch_size': 50,
    'xyu_samples_per_class': {'SUN': (200, 0, 400, 0),
                              'aPY': (200, 0, 400, 0),
                              'CUB': (200, 0, 400, 0),
                              'AWA2': (200, 0, 400, 0),
                              'FLO': (200, 0, 400, 0),
                              'AWA1': (200, 0, 400, 0)},
    'epochs': 100,
    'loss': 'l1',
    'auxiliary_data_source' : 'attributes',
    'lr_cls': 0.001,
    'dataset': 'CUB',
    'hidden_size_rule': {'resnet_features': (1560, 1660),
                        'attributes': (1450, 665),
                        'sentences': (1450, 665) },
    'latent_size': 64
}

# The training epochs for the final classifier, for early stopping,
# as determined on the validation spit

cls_train_steps = [
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 21},
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 30},
      {'dataset': 'SUN',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 22},
      {'dataset': 'SUN',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
      {'dataset': 'SUN',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'SUN',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'SUN',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'SUN',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 61},
      {'dataset': 'SUN',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 79},
      {'dataset': 'SUN',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 94},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 33},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 25},
      {'dataset': 'AWA1', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 40},
      {'dataset': 'AWA1', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 81},
      {'dataset': 'AWA1', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 89},
      {'dataset': 'AWA1', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 62},
      {'dataset': 'AWA1', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 56},
      {'dataset': 'AWA1', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 59},
      {'dataset': 'AWA1', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 100},
      {'dataset': 'AWA1', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 50},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'CUB',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'CUB',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'CUB',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'CUB',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'CUB',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'CUB',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'CUB',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'CUB',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 39},
      {'dataset': 'AWA2', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 44},
      {'dataset': 'AWA2', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
      {'dataset': 'AWA2', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 99},
      {'dataset': 'AWA2', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 100},
      {'dataset': 'AWA2', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 69},
      {'dataset': 'AWA2', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 79},
      {'dataset': 'AWA2', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 86},
      {'dataset': 'AWA2', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'aPY',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'aPY',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'aPY',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'aPY',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'aPY',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'aPY',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'aPY',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'aPY',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'aPY',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'aPY',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
      {'dataset': 'FLO',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'FLO',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'FLO',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'FLO',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'FLO',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'FLO',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'FLO',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'FLO',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'FLO',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'FLO',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67}
      ]

##################################
# change some hyperparameters here
##################################
hyperparameters['dataset'] = args.dataset
hyperparameters['num_shots']= args.num_shots
hyperparameters['generalized']= args.generalized

hyperparameters['cls_train_steps'] = [x['cls_train_steps']  for x in cls_train_steps
                                        if all([hyperparameters['dataset']==x['dataset'],
                                        hyperparameters['num_shots']==x['num_shots'],
                                        hyperparameters['generalized']==x['generalized'] ])][0]

print('***')
print(hyperparameters['cls_train_steps'])
if hyperparameters['generalized']:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (200, 0, 400, 0), 'SUN': (200, 0, 400, 0),
                                'aPY': (200, 0,  400, 0), 'AWA1': (200, 0, 400, 0),
                                'AWA2': (200, 0, 400, 0), 'FLO': (200, 0, 400, 0)}
    else:
        hyperparameters['samples_per_class'] = {'CUB': (200, 0, 200, 200), 'SUN': (200, 0, 200, 200),
                                                    'aPY': (200, 0, 200, 200), 'AWA1': (200, 0, 200, 200),
                                                    'AWA2': (200, 0, 200, 200), 'FLO': (200, 0, 200, 200)}
else:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, 200, 0), 'SUN': (0, 0, 200, 0),
                                                    'aPY': (0, 0, 200, 0), 'AWA1': (0, 0, 200, 0),
                                                    'AWA2': (0, 0, 200, 0), 'FLO': (0, 0, 200, 0)}
    else:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, 200, 200), 'SUN': (0, 0, 200, 200),
                                                    'aPY': (0, 0, 200, 200), 'AWA1': (0, 0, 200, 200),
                                                    'AWA2': (0, 0, 200, 200), 'FLO': (0, 0, 200, 200)}


model = Model(hyperparameters)
model.to(hyperparameters['device'])

"""
########################################
### load model where u left
########################################
saved_state = torch.load('./saved_models/CADA_trained.pth.tar')
model.load_state_dict(saved_state['state_dict'])
for d in model.all_data_sources_without_duplicates:
    model.encoder[d].load_state_dict(saved_state['encoder'][d])
    model.decoder[d].load_state_dict(saved_state['decoder'][d])
########################################
"""


losses = model.train_vae()

u,s,h,history = model.train_classifier()


if hyperparameters['generalized']==True:
    acc = [hi[2] for hi in history]
elif hyperparameters['generalized']==False:
    acc = [hi[1] for hi in history]

print(acc[-1])


state = {
            'state_dict': model.state_dict() ,
            'hyperparameters':hyperparameters,
            'encoder':{},
            'decoder':{}
        }
for d in model.all_data_sources:
    state['encoder'][d] = model.encoder[d].state_dict()
    state['decoder'][d] = model.decoder[d].state_dict()


torch.save(state, 'CADA_trained.pth.tar')
print('>> saved')































