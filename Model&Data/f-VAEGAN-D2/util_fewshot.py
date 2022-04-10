#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import h5py

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

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

        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        # train_loc = matcontent['train_loc'].squeeze() - 1
        # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),
                                                                                  self.attribute.size(1))
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        # else:
        #     self.train_feature = torch.from_numpy(feature[train_loc]).float()
        #     self.train_label = torch.from_numpy(label[train_loc]).long()
        #     self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
        #     self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
        #
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.test_seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.ntest_seen + self.ntest_unseen
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_att

