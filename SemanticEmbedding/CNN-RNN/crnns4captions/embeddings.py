'''Embed images prior to training text encoder.'''

# pylint: disable=no-member

import os
import argparse
from PIL import Image

import torch
import torchfile
import h5py
import numpy as np

from torchvision.transforms import transforms

from encoders import googlenet_feature_extractor

def embed(dataset_dir, image_dir, img_px, image_emb_dir, device, classes, train):
    '''Calculate the embeddings of images (given that the image encoder
    remains frozen during training) and save in .h5 format, where each image's
    embedding can be accessed at 'emb#'.'''

    image_dir = os.path.join(dataset_dir, image_dir)
    image_dir1 = "/data0/1259115645/CNN-RNN/flowers/text_c10"
    # image_emb_dir = os.path.join(dataset_dir, image_emb_dir)

    # get names of available classes
    avail_classes = []
    with open(os.path.join('/data0/1259115645/CNN-RNN/flowers', classes), 'r') as avcls:
        while True:
            line = avcls.readline()
            if not line:
                break
            avail_classes.append(line.strip())

    img_encoder = googlenet_feature_extractor().to(device).eval()
    pil2tensor = transforms.ToTensor()
    with torch.no_grad():
        j = 0
        max = 0
        min = 240
        for clas_dir in os.listdir(image_dir1):

            if clas_dir not in avail_classes: # if not instructed to meddle with class
                print("clas_dir not in avail_classes: ", clas_dir)
                continue

            # get file name of embeddings of class, e.g. bluh/001.Black_footed_Albatross.h5
            clas_embs_fn = os.path.join(image_emb_dir, clas_dir) + '.h5'
            print(j, clas_embs_fn)
            j += 1
            clas_ims = os.listdir(os.path.join(image_dir1, clas_dir))
            txt_path_list = []
            with h5py.File(clas_embs_fn, 'w') as h5fp:
                i = 0
                for clas_im in clas_ims:
                    if clas_im[0] == '.' or clas_im[-1] == '5':
                        continue
                    elif clas_im[-1] == 't':
                        clas_im = clas_im[:-3] + 'jpg'
                    txt_path_list.append(os.path.join("/data0/1259115645/CNN-RNN/flowers/text_c10", clas_dir, clas_im[:-3]) + "h5")
                    img = Image.open(os.path.join(image_dir, clas_im))
                    # img_name = os.path.splitext(clas_im)[0] # get name to keep corresp with text

                    img = pil2tensor(img.resize((img_px,)*2))
                    if img.size(0) != 3:
                        continue
                    embs = img_encoder(img.unsqueeze(0).cuda()).squeeze()

                    if device.startswith('cuda'):
                        embs = embs.cpu()
                    embs = embs.detach().numpy()
                    h5fp[str(i)] = embs
                    i += 1

            # txts = []
            #
            # for item in txt_path_list:
            #     print(item)
            #     with h5py.File(item, 'r') as item_con:
            #         keys = list(item_con.keys())  # make subscriptable
            #         cons = []
            #         for key in keys:
            #             con = item_con[key].value
            #             con = con.reshape((-1,1))
            #             if con.shape[0] < 201:
            #                 con = np.row_stack([con, np.zeros((201-con.shape[0],1))])
            #             elif con.shape[0] > 201:
            #                 con = con[:201]
            #             cons.append(con)
            #         cons = np.column_stack(cons)
            #         print("cons.shape: ", cons.shape)
            #         cons = cons.reshape((1,201,10))
            #     txts.append(cons)
            # txts = np.concatenate(txts,axis=0)
            # if max < np.max(txts):
            #     max = np.max(txts)
            # if min > np.min(txts):
            #     min = np.min(txts)
            # print("txts: ", txts.shape, max, min)
            # clas_sem_embs_fn = os.path.join("/data0/1259115645/CNN-RNN/flowers/text_c10_lazy", clas_dir) + '.h5'
            # with h5py.File(clas_sem_embs_fn, 'w') as h5fp:
            #     for k in range(txts.shape[0]):
            #         h5fp[str(k)] = txts[k]


# def transform(dataset_dir, image_dir, image_emb_dir, classes, clear):
#     '''Transform embeddings from .t7 to .h5'''
#
#     image_dir = os.path.join(dataset_dir, image_dir)
#     image_emb_dir = os.path.join(dataset_dir, image_emb_dir)
#
#     if not os.path.exists(image_emb_dir):
#         os.makedirs(image_emb_dir)
#
#     avail_classes = []
#     with open(os.path.join(dataset_dir, classes), 'r') as avcls:
#         while True:
#             line = avcls.readline()
#             if not line:
#                 break
#             avail_classes.append(line.strip())
#
#     for clas_embs in os.listdir(image_dir):
#         # get name of class
#         clas_name = os.path.splitext(clas_embs)[0]
#
#         if clas_name not in avail_classes:
#             continue # if not instructed to meddle with class
#
#         # get read and write filenames
#         clas_embs_fn = os.path.join(image_dir, clas_embs)
#         new_clas_embs_fn = os.path.join(image_emb_dir, clas_name + '.h5')
#         # n_images x 1024 x 10
#         embs = torchfile.load(clas_embs_fn)
#
#         with h5py.File(new_clas_embs_fn, 'w') as h5fp:
#             for img in range(embs.shape[0]):
#                 h5fp.create_dataset(f'img{img}', data=embs[img])
#
#         if clear:
#             os.system(f'rm {image_dir}/*.t7')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', required=True, type=str,
                        help='root directory of dataset')

    parser.add_argument('-i', '--image_dir', required=True, type=str,
                        help='directory of images')

    parser.add_argument('-ied', '--image_emb_dir', required=True, type=str,
                        help='directory to save embeddings')

    parser.add_argument('-px', '--img_px', default=224, type=int,
                        help='pixels for image to be resized to')

    parser.add_argument('-dev', '--device', default='cuda:0', type=str,
                        help='device to execute on')

    parser.add_argument('-c', '--clear', default=False, action='store_true',
                        help='whether to delete .t7 after transform')

    parser.add_argument('-cls', '--class_fn', type=str, required=True,
                        help='txt of classes to manipulate')

    parser.add_argument('--train', default=False, action='store_true',
                        help='whether')

    parser.add_argument('-emb', '--embed', default=False, action='store_true',
                        help='If set, creates embeddings, else transforms them')

    args = parser.parse_args()

    if args.embed:
        embed(args.dataset_dir, args.image_dir, args.img_px, args.image_emb_dir,
              args.device, args.class_fn, args.train)
    else:
        transform(args.dataset_dir, args.image_dir, args.image_emb_dir,
                  args.class_fn, args.clear)

# python embeddings.py -d /data0/1259115645/LSL/ZSLDB/CUB/CUB_200_2011 -i images -ied /data0/1259115645/CNN-RNN/cub/images_lazy -cls allclasses.txt -emb
# python embeddings.py -d /data0/1259115645/LSL/ZSLDB/FLO -i jpg -ied /data0/1259115645/CNN-RNN/flowers/images_lazy -cls allclasses.txt -emb