#author: sanath
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import util

class CLASSIFIER:
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, syn_feature, syn_label, _cuda, seen_classifier, unseen_classifier, _lr=0.001, _beta1=0.5, _nepoch=50, _batch_size=100, _hidden_size=512, netDec=None, dec_size=4096, dec_hidden_size=4096):
        self.train_X =  _train_X 
        self.train_Y = _train_Y 
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.seenclasses = data_loader.seenclasses
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.hidden_size = _hidden_size
        self.input_dim = _train_X.size(1)
        self.syn_feat = syn_feature
        self.syn_label = syn_label
        self.cuda = _cuda
        self.model = ODDetector(self.input_dim, self.hidden_size, self.nclass)
        self.netDec = netDec
        if self.netDec:
            self.netDec.eval()
            self.input_dim = self.input_dim + dec_size #4096
            self.input_dim += dec_hidden_size
            self.model = ODDetector(self.input_dim, self.hidden_size, self.nclass)
            self.train_X = self.compute_dec_out(self.train_X,self.input_dim)
            self.syn_feat = self.compute_dec_out(self.syn_feat,self.input_dim)
            self.test_unseen_feature = self.compute_dec_out(self.test_unseen_feature,self.input_dim)
            self.test_seen_feature = self.compute_dec_out(self.test_seen_feature,self.input_dim)

        self.seen_cls_model = seen_classifier.best_model
        self.unseen_cls_model = unseen_classifier.best_model
        self.model.apply(util.weights_init)
        self.criterion = HLoss()
        self.nll_criterion = nn.NLLLoss()
        self.logsoft = nn.LogSoftmax(dim=1)
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size).fill_(0) 
        self.lr = _lr 
        self.beta1 = _beta1
        # setup optimizer
        self.od_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.criterion = self.criterion.cuda()
            self.nll_criterion = self.nll_criterion.cuda()
            self.logsoft = self.logsoft.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.index_in_epoch_syn = 0
        self.ntrain = self.train_X.size()[0]

        self.acc_seen, self.acc_unseen, self.H = self.fit()
    
    def fit(self):
        best_seen = 0
        best_unseen = 0
        best_H = 0
        for epoch in range(self.nepoch):
            entr_seen = 0
            entr_unseen = 0
            hbsz = int(self.batch_size/2)
            batch_num = 0
            for i in range(0, self.ntrain, self.batch_size): 
                batch_num += 1
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(hbsz)
                batch_input2, batch_label2 = self.next_batch_syn(hbsz)
                self.input[:hbsz].copy_(batch_input)
                self.label[:hbsz].copy_(batch_label)                
                self.input[hbsz:].copy_(batch_input2)
                self.label[hbsz:].copy_(batch_label2)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                model_input = inputv
                pred = self.model(model_input)
                ## For seen classes, minimize entropy
                loss1 = self.criterion(pred[:hbsz], neg=True) + self.nll_criterion(self.logsoft(pred[:hbsz]),labelv[:hbsz])
                ## For unseen classes, maximize entropy
                loss2 = self.criterion(pred[hbsz:], neg=False)
                entropy_loss = loss1 + loss2
                entropy_loss.backward()
                entr_seen += self.criterion(pred[:hbsz], batch=True).sum()
                entr_unseen += self.criterion(pred[hbsz:], batch=True).sum()
                self.od_optimizer.step()

            # GZSL Evaluation using OD
            ent_thresh = entr_seen.data[0]/self.ntrain
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses, ent_thresh, seen_classes=True)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, ent_thresh, seen_classes=False) 
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen+1e-12)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                        
        return best_seen, best_unseen, best_H
    
    # Batch Sampler for seen data              
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        endt = self.index_in_epoch
        if endt > self.ntrain-batch_size:
            # shuffle the data and reset start counter
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            start = 0
            endt = start + batch_size
        return self.train_X[start:endt], self.train_Y[start:endt]

    # Fetch next batch for Synthetic features
    def next_batch_syn(self, batch_size):
        start = self.index_in_epoch_syn
        ntrain = self.syn_feat.size(0)
        self.index_in_epoch_syn += batch_size
        endt = self.index_in_epoch_syn
        if endt > ntrain-batch_size:
            # shuffle the data and reset start counter
            perm = torch.randperm(ntrain)
            self.syn_feat = self.syn_feat[perm]
            self.syn_label = self.syn_label[perm]
            start = 0
            endt = start + batch_size
        return self.syn_feat[start:endt], self.syn_label[start:endt]


    # GZSL eval
    def val_gzsl(self, test_X, test_label, target_classes, thresh, seen_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        entropy = []
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                test_Xv = Variable(test_X[start:end].cuda(), volatile=True)
            else:
                test_Xv = Variable(test_X[start:end], volatile=True)
            output = self.model(test_Xv) 
            entropy_batch = self.criterion(output, batch=True)
            # The following evaluation holds true as seen and unseen sets are validated separately.
            if seen_classes:
                pred = self.seen_cls_model(test_Xv)
            else:
                pred = self.unseen_cls_model(test_Xv)
                
            _, pred = torch.max(pred.data, 1)
            entropy.extend(entropy_batch.data.view(-1).cpu().numpy())
            predicted_label[start:end] = pred.cpu()
            start = end

        # The following threshold works as seen and unseen sets are validated separately.
        seen_mask = torch.Tensor(np.array(entropy)) < thresh
        if not seen_classes:
            seen_mask = 1 - seen_mask
        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, seen_mask)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes, mask):
        acc_per_class = 0
        test_label = util.map_label(test_label,target_classes)  # required to map for both classifiers
        for i in range(target_classes.size(0)):
            idx = (test_label == i)
            acc_per_class += torch.sum((test_label[idx]==predicted_label[idx])*mask[idx]) / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class


    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda(), volatile=True)
            else:
                inputX = Variable(test_X[start:end], volatile=True)
            feat1 = self.netDec(inputX)
            feat2 = self.netDec.getLayersOutDet()
            new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            start = end
        return new_test_X

    
class ODDetector(nn.Module):
    def __init__(self, input_dim, h_size, num_classes):
        super(ODDetector, self).__init__()
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(input_dim, h_size)
        self.fc2 = nn.Linear(h_size,h_size)
        self.classifier = nn.Linear(h_size, num_classes)
    
    def forward(self,x,center_loss=False):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        pred = self.classifier(h)
        return pred


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        self.logsoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, neg=True, batch=False):
        b = self.softmax(x) * self.logsoft(x)
        if batch:
            return -1.0 * b.sum(1)
        if neg:
            return -1.0 * b.sum()/x.size(0)
        else:
            return  b.sum()/x.size(0)
