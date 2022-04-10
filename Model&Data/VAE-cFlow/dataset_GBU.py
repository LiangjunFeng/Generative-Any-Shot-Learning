import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch
import h5py

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):

        if opt.dataset == 'ImageNet':
            self.read_matimagenet(opt)
        else:
            self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)

    def process_few_shot_train(self, data, attsplits, num):
        labels = data["labels"]
        from copy import deepcopy
        copy_labels = deepcopy(labels).reshape(-1,1)
        att = attsplits["att"]
        test_seen_loc = attsplits["test_seen_loc"]
        test_unseen_loc = attsplits["test_unseen_loc"]

        seen_classes = np.unique(np.ravel(labels)[test_seen_loc - 1]).tolist()
        copy_labels[test_seen_loc-1] = -1
        add_seen_index = []
        for i in seen_classes:
            # print(np.where(copy_labels == i))
            add_seen_index += np.where(copy_labels == i)[0].tolist()[0:num]
        # print(add_seen_index)
        trainval_loc = np.array(add_seen_index).reshape(-1, 1) + 1
        print(trainval_loc.shape)
        if trainval_loc.shape[0] < 1024:
             n = int(1024/trainval_loc.shape[0] + 1)
             trainval_loc = np.repeat(trainval_loc, n, axis=0)
        print(trainval_loc.shape)
        myLabel = {}
        myLabel["att"] = att
        myLabel["test_unseen_loc"] = test_unseen_loc
        myLabel["test_seen_loc"] = test_seen_loc
        myLabel["trainval_loc"] = trainval_loc
        return data, myLabel

    def process_few_shot_test(self, data, attsplits, num):
        labels = data["labels"]

        att = attsplits["att"]
        test_seen_loc = attsplits["test_seen_loc"]
        test_unseen_loc = attsplits["test_unseen_loc"]
        trainval_loc = attsplits["trainval_loc"]
        unseen_classes = np.unique(np.ravel(labels)[test_unseen_loc - 1]).tolist()
        # print(unseen_classes)
        add_unseen_index = []
        for i in unseen_classes:
            # print('*',i, np.where(labels.T == i),labels.T.shape)
            if (labels.shape[1] == 1):
                add_unseen_index += np.where(labels.T == i)[1].tolist()[0:num]
            else:
                add_unseen_index += np.where(labels == i)[1].tolist()[0:num]
        # print(len(add_unseen_index))
        trainval_loc = np.row_stack([trainval_loc, np.array(add_unseen_index).reshape(-1, 1) + 1])
        # print(add_unseen_index)
        for i in add_unseen_index:
            # print('&',i, np.where(test_unseen_loc == i + 1))
            ind = np.where(test_unseen_loc == i + 1)[0][0]
            # print(ind)
            test_unseen_loc = np.delete(test_unseen_loc, ind, 0)

        myLabel = {}
        myLabel["att"] = att
        myLabel["test_unseen_loc"] = test_unseen_loc
        myLabel["test_seen_loc"] = test_seen_loc
        myLabel["trainval_loc"] = trainval_loc
        return data, myLabel

    def read_matimagenet(self, opt):
        if opt.preprocessing:

            print('MinMaxScaler...')

            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            print('load train feature')
            matcontent = h5py.File('/media/guyuchao/data/gyc/PycharmProject/Zero-shot-cls/VAE_cFlow/data/ImageNet/lp500/lp500.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
            print('load test feature')

        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = np.array(matcontent['w2v']).T
        l2normalizer = preprocessing.Normalizer(norm='l2')
        self.attribute = l2normalizer.fit_transform(self.attribute)

        self.train_feature = torch.from_numpy(feature).float()

        self.train_label = torch.from_numpy(label).long()

        self.test_seen_feature = torch.from_numpy(feature_val).float()

        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()

        self.test_unseen_label = torch.from_numpy(label_unseen).long()

        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))

        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses]
        self.test_att = self.attribute[self.unseenclasses]


    def read_matdataset(self, opt):
        matcontent1 = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        if opt.num_shots > 0:
            if opt.few_train:
                matcontent1, matcontent = self.process_few_shot_train(matcontent1, matcontent, opt.num_shots)
            else:
                matcontent1, matcontent = self.process_few_shot_test(matcontent1, matcontent, opt.num_shots)
        feature = matcontent1['features'].T
        label = matcontent1['labels'].astype(int).squeeze() - 1

        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        #train_loc = matcontent['train_loc'].squeeze() - 1
        #val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        if not opt.validation:
            scaler = preprocessing.MinMaxScaler()
            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            #_train_feature = feature[trainval_loc]
            #_test_seen_feature = feature[test_seen_loc]
            #_test_unseen_feature = feature[test_unseen_loc]
            self.train_feature = torch.from_numpy(_train_feature).float()

            mx = self.train_feature.max()

            self.train_feature.mul_(1 / mx)
            self.train_label = torch.from_numpy(label[trainval_loc]).long()
            self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
            self.test_unseen_feature.mul_(1 / mx)
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
            self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()

            #dsa
            self.test_seen_feature.mul_(1 / mx)
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        # else:
        #     self.train_feature = torch.from_numpy(feature[train_loc]).float()
        #     self.train_label = torch.from_numpy(label[train_loc]).long()
        #     self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
        #     self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.test_seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        # self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_label = map_label(self.train_label, self.seenclasses)
        self.mytest_unseen_label = self.test_unseen_label
        self.mytest_seen_label = self.test_seen_label
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att  = self.attribute[self.unseenclasses].numpy()



class FeatDataLayer(object):
    def __init__(self, label, feat_data,  opt):
        """Set the roidb to be used by this layer during training."""
        #self._roidb = roidb
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()
        self._epoch = 0
    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def forward(self):
        new_epoch = False
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'newEpoch': new_epoch, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs


