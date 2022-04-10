from scipy.io import loadmat, savemat
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import sys
import random

random.seed(7203)
torch.manual_seed(7203)

# CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_reg.py FLO > flo_data.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_reg.py AWA2 > awa2_data.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_reg.py CUB > cub_data.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_reg.py SUN > sun_data.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_reg.py aPY > apy_data.log 2>&1 &


path = "/data0/1259115645/LSL/ZSLDB/"
savepath = "/data0/1259115645/LSL/data/"
# path = "/Volumes/文档/ZSLDB/"
# aPY, AWA2, CUB, FLO, SUN
benchmark = sys.argv[1]
# benchmark = "CUB"
print(benchmark)

# ================================文件准备=============================

att_split = loadmat(path + benchmark + "/att_splits.mat")
data = loadmat(path + benchmark + "/res101.mat")

if benchmark in ["AWA2", "FLO", "SUN"]:
    data['image_files'] = np.array(
        list(map(lambda x: path + benchmark + '/' + '/'.join(x[0][0].split("/")[7:]), data['image_files'])))
elif benchmark == "CUB":
    data['image_files'] = np.array(
        list(map(lambda x: path + benchmark + '/' + '/'.join(x[0][0].split("/")[6:]), data['image_files'])))
elif benchmark == "aPY":
    def path_process_aPY(x):
        length = len(x[0][0].split("/"))
        if length == 12:
            return path + benchmark + '/' + '/'.join(x[0][0].split("/")[8:])
        else:
            return path + benchmark + '/' + '/'.join(x[0][0].split("/")[7:])


    data['image_files'] = np.array(list(map(path_process_aPY, data['image_files'])))

data['labels'] = np.ravel(data['labels'] - 1)
data['attribute'] = att_split['att']
att = data['attribute'].T[data['labels']].squeeze()

# =======GZSL设定索引============
data['test_seen_index'] = np.ravel(att_split['test_seen_loc'] - 1).tolist()
data['test_unseen_index'] = np.ravel(att_split['test_unseen_loc'] - 1).tolist()
data['train_seen_index'] = list(
    set(np.arange(len(data['image_files'])).tolist()) - set(data['test_unseen_index']) - set(data['test_seen_index']))

unseenclasses = np.unique(data['labels'][data['test_unseen_index']]).tolist()
seenclasses = np.unique(data['labels'][data['test_seen_index']]).tolist()
data['unseenclasses'] = unseenclasses
data['seenclasses'] = seenclasses

# =======GFSL设定索引============
data['fsl_test_seen_index'] = data['test_seen_index']
fsl_train_unseen_index = np.zeros([np.unique(data['labels']).shape[0], 10]).astype(int)
unseenlabels = set(data['test_unseen_index'])

for i in unseenclasses:
    temp = np.where(data['labels'] == i)[0][:10]
    fsl_train_unseen_index[i] = temp
    unseenlabels = unseenlabels - set(temp.tolist())
data['fsl_test_unseen_index'] = list(unseenlabels)
data['fsl_train_unseen_index'] = fsl_train_unseen_index
data['fsl_train_seen_index'] = data['train_seen_index']

#=======center loss============
class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, bound = 1,beta = 0.9):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.bound = bound
        self.beta = beta

    def forward(self, similarity, dis_similarity):
        return torch.max(self.bound + self.beta*similarity - (1-self.beta)*dis_similarity, torch.tensor(0.0).cuda())


def Other_label(labels,num_classes):
    index=torch.randint(num_classes, (labels.shape[0],)).to(labels.device)
    other_labels=labels+index
    other_labels[other_labels >= num_classes]=other_labels[other_labels >= num_classes]-num_classes
    return other_labels


class TripCenterLoss_margin(nn.Module):

    def __init__(self, num_classes=10, feat_dim=312, use_gpu=True):
        super(TripCenterLoss_margin, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    def forward(self, x, labels,margin, incenter_weight):
        other_labels = Other_label(labels, self.num_classes)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]
        other_labels = other_labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask_other = other_labels.eq(classes.expand(batch_size, self.num_classes))
        dist_other = distmat[mask_other]
        loss = torch.max(margin+incenter_weight*dist-(1-incenter_weight)*dist_other,torch.tensor(0.0).cuda()).sum() / batch_size
        return loss

class TripCenterLoss_min_margin(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(TripCenterLoss_min_margin, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    def forward(self, x, labels,margin, incenter_weight):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]

        other=torch.FloatTensor(batch_size,self.num_classes-1).cuda()
        for i in range(batch_size):
            other[i]=(distmat[i,mask[i,:]==0])

        dist_min,_=other.min(dim=1)
        loss = torch.max(margin+incenter_weight*dist-(1-incenter_weight)*dist_min,torch.tensor(0.0).cuda()).sum() / batch_size
        return loss


# class CenterLoss(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#
#     def forward(self, x, labels, att, margin=150, incenter_weight=0.5):
#         other_labels = Other_label(labels, self.num_classes)
#         centers = att
#         batch_size = x.size(0)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                   torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#         distmat.addmm_(1, -2, x, centers.t())
#         classes = torch.arange(self.num_classes).long()
#         classes = classes.cuda()
#         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask = labels.eq(classes.expand(batch_size, self.num_classes))
#         dist = distmat[mask]
#         other_labels = other_labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask_other = other_labels.eq(classes.expand(batch_size, self.num_classes))
#         dist_other = distmat[mask_other]
#         loss = torch.max(margin+incenter_weight*dist-(1-incenter_weight)*dist_other,torch.tensor(0.0).cuda()).sum() / batch_size
#         return loss


# ==============================构造数据集==============================

def map_label(label, classes):
    mapped_label = np.zeros(label.shape)
    for i in range(classes.shape[0]):
        mapped_label[label == classes[i]] = i
    return mapped_label


traindata_files = data['image_files'][data['train_seen_index']]
trainlabel = data['labels'][data['train_seen_index']]
trainatt = att[data['train_seen_index']]
trainlabel = map_label(trainlabel, np.unique(trainlabel))
loc_trainatt = np.random.random((len(data['seenclasses']), trainatt.shape[1]))
for i in range(len(data['seenclasses'])):
    loc_trainatt[i] = trainatt[np.ravel(trainlabel).tolist().index(i)]
loc_trainatt = torch.from_numpy(loc_trainatt).cuda()

alldata_files = data['image_files']
alldataatt = att
alllabel = data['labels']


class dataset(Dataset):
    def __init__(self, data_files, label, att, transform):
        super().__init__()
        self.data_files = data_files
        self.label = label
        self.att = att
        self.transform = transform

    def __getitem__(self, index):
        data = Image.open(self.data_files[index])
        data = self.transform(data)
        if data.shape[0] == 1:
            data = data.repeat([3, 1, 1])
        elif data.shape[0] == 4:
            data = data[:3, :, :]
        data = normalize(data)
        label = self.label[index]
        att = self.att[index]
        return data, label / 1.0, att

    def __len__(self):
        return self.data_files.shape[0]


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()

    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
}

normalize = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = dataset(traindata_files, trainlabel, trainatt, data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

all_dataset = dataset(alldata_files, alllabel, alldataatt, data_transforms['val'])
all_loader = DataLoader(all_dataset, batch_size=16, shuffle=False, num_workers=4)


# #==============================ResNet101特征提取==============================


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


def extract_feature(model, exact_list=['avgpool']):
    model.eval()
    myexactor = FeatureExtractor(model, exact_list)

    feature_list = []
    for data in tqdm(all_loader):
        inputs, _, _ = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        feature = myexactor(inputs)[0]
        feature = feature.view(feature.shape[0], feature.shape[1])
        feature_list.append(feature.detach().cpu().numpy())

    features = np.row_stack(feature_list)
    print(features.shape)
    return features


def train_model(model, criterion, nclasses, center_criterion, optimizer, scheduler, lambd, num_epochs):
    for epoch in range(num_epochs):
        if epoch in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]:
            data['features'] = extract_feature(model_ft, exact_list=['avgpool'])
            savemat(savepath + benchmark + "/res101_finetune_reg_" + str(epoch) + "epo.mat",
                    {'features': data['features'].T,
                     'image_files': data['image_files'],
                     'labels': (data['labels'] + 1).T,
                     'seenclasses': data['seenclasses'],
                     'unseenclasses': data['unseenclasses'],
                     'test_unseen_index': data['test_unseen_index'],
                     'test_seen_index': data['test_seen_index'],
                     'train_seen_index': data['train_seen_index'],
                     'attribute': data['attribute'],
                     'fsl_test_unseen_index': data['fsl_test_unseen_index'],
                     'fsl_test_seen_index': data['fsl_test_seen_index'],
                     'fsl_train_unseen_index': data['fsl_train_unseen_index'],
                     'fsl_train_seen_index': data['fsl_train_seen_index']
                     })
            print(path + benchmark + "/res101_finetune_reg_" + str(epoch) + "epo.mat" + " is saved!")

        model.train()
        running_loss = 0.0
        pre_loss = 0.0
        mse_loss = 0.0
        running_corrects = 0
        for inputs, labels, att in tqdm(train_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.long().cuda()
                att = att.float().cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs1 = outputs[:,:nclasses]
            outputs2 = outputs[:,nclasses:]
            _, preds = torch.max(outputs1, 1)

            loss1 = criterion(outputs1, labels)
            # similarity = torch.mean(torch.cosine_similarity(outputs2, att, dim=1)).cuda()
            similarity = torch.mean(torch.norm(outputs2-att, p=2, dim=1)).cuda()
            other_labels = Other_label(labels, nclasses)
            other_att = loc_trainatt[other_labels].squeeze()
            # dis_similarity = torch.mean(torch.cosine_similarity(outputs2.float(), other_att.float() ,dim=1)).cuda()
            dis_similarity = torch.mean(torch.norm(outputs2.float()-other_att.float(), p=2, dim=1)).cuda()
            loss2 = center_criterion(similarity, dis_similarity)

            loss = loss1 + lambd*loss2
            loss.backward()
            optimizer.step()

            pre_loss += loss1.item() * inputs.size(0)
            mse_loss += lambd*loss2.item() * inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        epoch_pre_loss = pre_loss / len(train_dataset)
        epoch_mse_loss = mse_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print('{} epoch loss: {:.4f}, epoch pre loss: {:.4f}, epoch center loss: {:.6f}, acc: {:.4f}'
              .format(epoch + 1, epoch_loss, epoch_pre_loss, epoch_mse_loss,  epoch_acc))
        print()
    return model


nclasses = len(np.unique(trainlabel))
lr = 0.01
if benchmark in ["FLO", "SUN"]:
    bound = 1; lambd = 0.01; beta = 0.9
elif benchmark in ["CUB"]:
    bound = 0.01; lambd = 0.1; beta = 0.99
elif benchmark in ["aPY"]:
    bound = 0.01; lambd = 0.001; beta = 0.99
elif benchmark in ["awa2"]:
    bound = 0.1; lambd = 0.1; beta = 0.9
# flo、sun bound 1 lambd 0.01 beta 0.9
# cub bound 0.01 lambd 0.1 beta 0.99
# apy bound 0.01 lambd 0.001 beta 0.99
# awa2 bound 0.1 lambd 0.1 beta 0.9

print(bound, lambd, beta)
model_ft = models.resnet101(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, nclasses+att.shape[1])
print(nclasses, att.shape[1])
criterion = nn.CrossEntropyLoss()
center_criterion = CenterLoss(nclasses, feat_dim=att.shape[1], bound = bound, beta = beta)

if torch.cuda.is_available():
    model_ft = model_ft.cuda()
    criterion = criterion.cuda()
    center_criterion = center_criterion.cuda()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, nclasses, center_criterion, optimizer_ft, exp_lr_scheduler, lambd, num_epochs=81)

































































