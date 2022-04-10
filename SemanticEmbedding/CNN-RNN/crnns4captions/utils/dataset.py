'''CUB-200-2011 Pytorch dataset.'''

# pylint: disable=no-member
# pylint: disable=not-callable

__all__ = ['CUBDataset', 'CUBDatasetLazy']

import os
import string
import torch
# import torchvision.transforms as transforms
import torchfile
import h5py


class CUBDatasetLazy(torch.utils.data.Dataset):

    # pylint: disable=abstract-method
    # pylint: disable=too-many-instance-attributes

    '''CUB-200-2011 dataset.'''
    def __init__(self, dataset_dir: str, avail_class_fn: str, image_dir: str,
                 text_dir: str, device='cuda:0', **kwargs):

        # pylint: disable=too-many-arguments

        '''Initialize dataset. Note that non lazy dirs are expected,
        they will be used wherever necessary.'''

        super().__init__()

        # alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
        self.vocab_len = 70

        self.avail_classes = [] # classes to read from
        with open(os.path.join(dataset_dir, avail_class_fn), 'r') as avcls:
            while True:
                line = avcls.readline()
                if not line:
                    break
                self.avail_classes.append(line.strip())

        self.dataset_dir = dataset_dir
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.device = device

        # number of classes to return
        if 'minibatch_size' in kwargs and kwargs['minibatch_size'] > 1:
            self.minibatch_size = min(kwargs['minibatch_size'], len(self.avail_classes))
        else:
            self.minibatch_size = len(self.avail_classes)

    def get_captions(self):
        '''Creates generator that yields one class' captions at a time in a
        `torch.Tensor` of size `images`x`10`x`vocabulary_size`x`caption_max_size`.
        Label is also returned.'''

        for clas in self.avail_classes:
            lbl = int((clas.split('.')[0]).split('_')[1])

            txt_fn = os.path.join(self.dataset_dir, self.text_dir, clas + '.t7')
            txt_np = torchfile.load(txt_fn)
            txt_t = self.process_text(txt_np)

            yield txt_t, lbl

    def get_images(self):
        '''Creates generator that yields one class' image embeddings
        at a time in a `torch.Tensor`. Label is also returned.'''

        for clas in self.avail_classes:
            lbl = int((clas.split('.')[0]).split('_')[1])

            imgs_fn = os.path.join(self.dataset_dir, self.image_dir, clas + '.t7')
            imgs_np = torchfile.load(imgs_fn)
            # the original image is used during inference -> index 0
            imgs_t = torch.tensor(imgs_np[..., 0], dtype=torch.float, device=self.device)

            yield imgs_t, lbl


    def get_next_minibatch(self, n_txts=1):
        '''Get next training batch as suggested in
        `Learning Deep Representations of Fine-Grained Visual Descriptions`, i.e.
        one image's embeddings with `n_txts` matching descriptions is returned from
        every class along with their labels.'''

        assert 1 <= n_txts <= 10

        imgs = torch.empty(self.minibatch_size, 1024, device=self.device)
        txts = torch.empty(self.minibatch_size, n_txts, self.vocab_len,
                           201, device=self.device)
        lbls = torch.empty(self.minibatch_size, dtype=int, device=self.device)

        rand_class_ind = torch.randperm(len(self.avail_classes))[:self.minibatch_size]
        for i, class_ind in enumerate(rand_class_ind):
            clas = self.avail_classes[class_ind]

            lbl = int((clas.split('.')[0]).split('_')[1])

            img_fn = os.path.join(self.dataset_dir, self.image_dir + '_lazy', clas + '.h5')
            with h5py.File(img_fn, 'r') as h5fp:
                # print("h5fp: ", h5fp)
                # pick an image from the class at rand
                # print("len(h5fp): ", len(h5fp))
                rand_img = str(torch.randint(len(h5fp), (1,)).item())
                # print("rand_img: ", rand_img)
                # pick a crop at rand
                rand_crop = torch.randint(10, (1,)).item()
                imgs[i] = torch.tensor(h5fp[rand_img][..., rand_crop], device=self.device)

            txt_fn = os.path.join(self.dataset_dir, self.text_dir + '_lazy', clas + '.h5')
            with h5py.File(txt_fn, 'r') as h5fp:
                # get n_txts random texts
                rand_txts = torch.randperm(10)[:n_txts]
                # reshape because process text expects 3d
                txts[i] = self.process_text(h5fp[rand_img][..., rand_txts].reshape(1, 201, len(rand_txts)))

            lbls[i] = lbl

        return imgs, txts.squeeze(), lbls

    def process_text(self, text):
        '''Transform np array of ascii codes to one-hot sequence.'''

        ohvec = torch.zeros(text.shape[0], text.shape[2], self.vocab_len, 
                            text.shape[1], device=self.device)

        for corr_img in range(text.shape[0]):
            for cap in range(text.shape[2]):
                for tok in range(text.shape[1]):
                    # -1 because of lua indexing
                    ohvec[corr_img, cap, int(text[corr_img, tok, cap])-1, tok] = 1

        return ohvec
