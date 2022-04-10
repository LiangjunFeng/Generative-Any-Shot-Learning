'''Fetch and evaluate best text encoder.'''

# pylint: disable=no-member

import os
import argparse

import torch
import numpy as np

from utils import CUBDatasetLazy, Fvt, model_name, get_hyperparameters_from_entry
from encoders import HybridCNN
from scipy.io import savemat

def test_best():
    '''Main'''

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', required=True, type=str,
                        help='dataset root directory')

    parser.add_argument('-avc', '--avail_class_fn', required=True, type=str,
                        help='txt containing classes used')

    parser.add_argument('-i', '--image_dir', required=True, type=str,
                        help='directory of images w.r.t dataset directory')

    parser.add_argument('-t', '--text_dir', required=True, type=str,
                        help='directory of descriptions w.r.t detaset directory')


    parser.add_argument('-md', '--model_dir', type=str, required=True,
                        help='where to retrieve model\'s parameters')

    parser.add_argument('-dev', '--device', type=str, default='cuda:0',
                        help='device to execute on')

    parser.add_argument('-s', '--summary', type=str, help='where resulting metrics are saved')

    parser.add_argument('-c', '--clear', default=False, action='store_true',
                        help='whether to clean models folder')

    args = parser.parse_args()

    with open(args.summary, 'r') as fp:
        best = (-1, '')
        while True:
            row = fp.readline()
            if not row:
                break
            score_ind = row.index(',')
            if best[0] < float(row[:score_ind]):
                best = (float(row[:score_ind]), row[score_ind+1:])

    margs = get_hyperparameters_from_entry(best[1])
    setattr(margs, 'model_dir', args.model_dir)

    evalset = CUBDatasetLazy(dataset_dir=args.dataset_dir, avail_class_fn=args.avail_class_fn,
                             image_dir=args.image_dir, text_dir=args.text_dir, device=args.device)

    txt_encoder = HybridCNN(vocab_dim=evalset.vocab_len, conv_channels=margs.conv_channels,
                            conv_kernels=margs.conv_kernels, conv_strides=margs.conv_strides,
                            rnn_num_layers=margs.rnn_num_layers, rnn_bidir=margs.rnn_bidir,
                            rnn_hidden_size=margs.rnn_hidden_size//(1+int(margs.rnn_bidir)),
                            lstm=margs.lstm).to(args.device).eval()
    txt_encoder.load_state_dict(torch.load(model_name(margs), map_location=args.device))



    print(len(evalset.avail_classes))
    print(evalset.avail_classes)
    # mean_txt_embs = torch.empty(len(evalset.avail_classes), 1024, device=args.device)
    txt_embs = []
    txt_label = []
    with torch.no_grad():
        for i, (captions, _lbl) in enumerate(evalset.get_captions()):
            print(i, captions.view(-1, *captions.size()[-2:]).cpu().detach().numpy().shape)
            class_sample_feature = txt_encoder(captions.view(-1, *captions.size()[-2:]))
            txt_embs.append(class_sample_feature.cpu().detach().numpy())
            txt_label += [i]*class_sample_feature.shape[0]
            # mean_txt_embs[i] = txt_encoder(captions.view(-1, *captions.size()[-2:])).mean(dim=0)
    txt_embs = np.row_stack(txt_embs)
    txt_label = np.array(txt_label)
    print(txt_embs.shape, txt_label.shape)
    savemat("att_sample_naive.mat",{"att":txt_embs,"label":txt_label})

    print(len(evalset.avail_classes))
    print(evalset.avail_classes)
    mean_txt_embs = torch.empty(len(evalset.avail_classes), 1024, device=args.device)
    with torch.no_grad():
        for i, (captions, _lbl) in enumerate(evalset.get_captions()):
            print(i, captions.view(-1, *captions.size()[-2:]).cpu().detach().numpy().shape)
            mean_txt_embs[i] = txt_encoder(captions.view(-1, *captions.size()[-2:])).mean(dim=0)
    print(mean_txt_embs.cpu().detach().numpy().shape)
    print(mean_txt_embs.cpu().detach().numpy())
    # np.save("att_naive.npy", mean_txt_embs.cpu().detach().numpy())

    corr, outa = 0, 0
    for i, (img_embs, _lbl) in enumerate(evalset.get_images()):
        preds = Fvt(img_embs, mean_txt_embs).max(dim=1)[1]
        corr += (preds == i).sum().item()
        outa += len(preds)

    print(f'Test set Accuracy={corr/outa*100:5.2f}%')

    if args.clear:
        os.system(f'rm -rf {margs.model_dir}/*.pt')
        torch.save(txt_encoder.state_dict(), model_name(margs))

if __name__ == '__main__':
    test_best()
