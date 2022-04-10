import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from data_loader import DATA_LOADER as dataloader
import final_classifier as classifier
import models
import random
import torch.autograd as autograd
from torch.autograd import Variable
import classifier
import classifier2
import time
import numpy as np

class Model(nn.Module):
    def __init__(self, hyperparameters):
        super(Model, self).__init__()
        self.dataset = hyperparameters['dataset']
        self.few_train =  hyperparameters['few_train']
        self.num_shots = hyperparameters['num_shots']
        self.generalized = hyperparameters['generalized']

        self.dataroot = hyperparameters['dataroot']
        self.image_embedding = hyperparameters['image_embedding']
        self.class_embedding = hyperparameters['class_embedding']
        self.syn_num = hyperparameters['syn_num']
        self.preprocessing = hyperparameters['preprocessing']
        self.standardization = hyperparameters['standardization']
        self.validation = hyperparameters['validation']
        self.workers = hyperparameters['workers']
        self.batch_size = hyperparameters['batch_size']
        self.resSize = hyperparameters['resSize']
        self.attSize = hyperparameters['attSize']
        self.nz = hyperparameters['nz']
        self.ngh = hyperparameters['ngh']
        self.ndh = hyperparameters['ndh']
        self.nepoch = hyperparameters['nepoch']
        self.critic_iter = hyperparameters['critic_iter']
        self.lambda1 = hyperparameters['lambda1']
        self.cls_weight = hyperparameters['cls_weight']
        self.lr = hyperparameters['lr']
        self.classifier_lr = hyperparameters['classifier_lr']
        self.beta1 = hyperparameters['beta1']
        self.cuda = hyperparameters['cuda']
        self.ngpu = hyperparameters['ngpu']
        self.print_every = hyperparameters['print_every']
        self.start_epoch = hyperparameters['start_epoch']
        self.manualSeed = hyperparameters['manualSeed']
        self.nclass_all = hyperparameters['nclass_all']
        self.begin_time = time.time()
        self.run_time1 = 0
        self.run_time2 = 0


        if self.manualSeed is None:
            self.manualSeed = random.randint(1, 10000)
        if self.cuda:
            torch.cuda.manual_seed_all(self.manualSeed)
        random.seed(self.manualSeed)

        self.data = dataloader(hyperparameters)

        self.netG = models.MLP_G(hyperparameters)
        print(self.netG)
        self.netD = models.MLP_CRITIC(hyperparameters)
        print(self.netD)

        # classification loss, Equation (4) of the paper
        self.cls_criterion = nn.NLLLoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.input_res = torch.FloatTensor(self.batch_size, self.resSize)
        self.input_att = torch.FloatTensor(self.batch_size, self.attSize)
        self.noise = torch.FloatTensor(self.batch_size, self.nz)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1
        self.input_label = torch.LongTensor(self.batch_size)

        self.best_s = 0
        self.best_u = 0
        self.best_h = 0
        self.best_t = 0

        if self.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.input_res = self.input_res.cuda()
            self.noise, self.input_att = self.noise.cuda(), self.input_att.cuda()
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()
            self.cls_criterion.cuda()
            self.input_label = self.input_label.cuda()

        self.pretrain_cls = classifier.CLASSIFIER(self.data.train_feature,
                                                  self.map_label(self.data.train_label, self.data.seenclasses),
                                                  self.data.seenclasses.size(0), self.resSize, self.cuda, 0.001, 0.5,
                                                  50, 100)

        for p in self.pretrain_cls.model.parameters():
            p.requires_grad = False

    def sample(self):
        batch_feature, batch_label, batch_att = self.data.next_batch(self.batch_size)
        self.input_res.copy_(batch_feature)
        self.input_att.copy_(batch_att)
        self.input_label.copy_(self.map_label(batch_label, self.data.seenclasses))

    def map_label(self, label, classes):
        mapped_label = torch.LongTensor(label.size())
        for i in range(classes.size(0)):
            mapped_label[label == classes[i]] = i
        return mapped_label

    def calc_gradient_penalty(self, netD, real_data, fake_data, input_att):
        # print real_data.size()
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates, Variable(input_att))

        ones = torch.ones(disc_interpolates.size())
        if self.cuda:
            ones = ones.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda1
        return gradient_penalty

    def train_fclswgan(self):
        for epoch in range(self.nepoch):
            mean_lossD = 0
            mean_lossG = 0
            for i in range(0, self.data.ntrain, self.batch_size):
                ############################
                # (1) Update D network: optimize WGAN-GP objective, Equation (2)
                ###########################
                for p in self.netD.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                for iter_d in range(self.critic_iter):
                    self.sample()
                    self.netD.zero_grad()
                    # train with realG
                    # sample a mini-batch
                    input_resv = Variable(self.input_res)
                    input_attv = Variable(self.input_att)

                    criticD_real = self.netD(input_resv, input_attv)
                    criticD_real = criticD_real.mean()
                    criticD_real.backward(self.mone)

                    # train with fakeG
                    self.noise.normal_(0, 1)
                    noisev = Variable(self.noise)
                    fake = self.netG(noisev, input_attv)

                    criticD_fake = self.netD(fake.detach(), input_attv)
                    criticD_fake = criticD_fake.mean()
                    criticD_fake.backward(self.one)

                    # gradient penalty
                    gradient_penalty = self.calc_gradient_penalty(self.netD, self.input_res, fake.data, self.input_att)
                    gradient_penalty.backward()

                    Wasserstein_D = criticD_real - criticD_fake
                    D_cost = criticD_fake - criticD_real + gradient_penalty
                    self.optimizerD.step()

                ############################
                # (2) Update G network: optimize WGAN-GP objective, Equation (2)
                ###########################
                for p in self.netD.parameters():  # reset requires_grad
                    p.requires_grad = False  # avoid computation

                self.netG.zero_grad()
                input_attv = Variable(self.input_att)
                self.noise.normal_(0, 1)
                noisev = Variable(self.noise)
                fake = self.netG(noisev, input_attv)
                criticG_fake = self.netD(fake, input_attv)
                criticG_fake = criticG_fake.mean()
                G_cost = -criticG_fake
                # classification loss
                c_errG = self.cls_criterion(self.pretrain_cls.model(fake), Variable(self.input_label))
                errG = G_cost + self.cls_weight * c_errG
                errG.backward()
                self.optimizerG.step()

            mean_lossG /= self.data.ntrain / self.batch_size
            mean_lossD /= self.data.ntrain / self.batch_size
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
                  % (epoch, self.nepoch, D_cost.data.item(), G_cost.data.item(), Wasserstein_D.data.item(),
                     c_errG.data.item()))

            self.train_classifier()

    def generate_syn_feature(self, netG, classes, attribute, num):
        nclass = classes.size(0)
        syn_feature = torch.FloatTensor(nclass * num, self.resSize)
        syn_label = torch.LongTensor(nclass * num)
        syn_att = torch.FloatTensor(num, self.attSize)
        syn_noise = torch.FloatTensor(num, self.nz)
        if self.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()

        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            with torch.no_grad():
                output = netG(Variable(syn_noise), Variable(syn_att))
            syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i * num, num).fill_(iclass)

        return syn_feature, syn_label

    def obtain_unseen_data(self, num):
        unseenlabel = self.data.test_unseen_label[self.data.test_unseen_label == self.data.unseenclasses[0]][:num]
        unseendata = self.data.test_unseen_feature[self.data.test_unseen_label == self.data.unseenclasses[0]][:num]

        for i in range(1, self.data.unseenclasses.size(0)):
            unseenlabel = torch.cat((unseenlabel, self.data.test_unseen_label[
                                                      self.data.test_unseen_label == self.data.unseenclasses[i]][:num]),
                                    0)
            unseendata = torch.cat((unseendata, self.data.test_unseen_feature[
                                                    self.data.test_unseen_label == self.data.unseenclasses[i]][:num]),
                                   0)

        return unseendata, unseenlabel

    def train_classifier(self):
        self.netG.eval()
        # Generalized zero-shot learning
        if self.generalized:
            syn_feature, syn_label = self.generate_syn_feature(self.netG, self.data.unseenclasses, self.data.attribute, self.syn_num)
            train_X = torch.cat((self.data.train_feature, syn_feature), 0)
            train_Y = torch.cat((self.data.train_label, syn_label), 0)

            from scipy.io import savemat
            print(syn_feature.cpu().detach().numpy().shape, syn_label.cpu().detach().numpy().shape, self.data.train_feature.cpu().detach().numpy().shape,
                  self.data.train_label.cpu().detach().numpy().shape, self.data.test_unseen_feature.cpu().detach().numpy().shape, self.data.test_unseen_label.cpu().detach().numpy().shape,
                  self.data.test_seen_feature.cpu().detach().numpy().shape, self.data.test_seen_label.cpu().detach().numpy().shape)
            mydata = {"train_unseen_data": syn_feature.cpu().detach().numpy(),
                      "train_unseen_label": syn_label.cpu().detach().numpy(),
                      "train_seen_data": self.data.train_feature.cpu().detach().numpy(),
                      "train_seen_label": self.data.train_label.cpu().detach().numpy(),
                      "test_unseen_data":  self.data.test_unseen_feature.cpu().detach().numpy(),
                      "test_unseen_label": self.data.test_unseen_label.cpu().detach().numpy(),
                      "test_seen_data": self.data.test_seen_feature.cpu().detach().numpy(),
                      "test_seen_label": self.data.test_seen_label.cpu().detach().numpy()}
            savemat("fclswgan_data.mat", mydata)
            print("fclswgan_data.mat is saved!")

            nclass = self.nclass_all
            cls = classifier2.CLASSIFIER(train_X, train_Y, self.data, nclass, self.cuda, self.classifier_lr, 0.5, 25,
                                         self.syn_num, True)
            if self.best_h < cls.H:
                self.best_h = cls.H
                self.best_u = cls.acc_unseen
                self.best_s = cls.acc_seen
                syn_feature, syn_label = self.generate_syn_feature(self.netG, self.data.unseenclasses,
                                                                   self.data.attribute, 500)

                np.save("./fclswgan_feat.npy", syn_feature.data.cpu().numpy())
                np.save("./fclswgan_label.npy", syn_label.data.cpu().numpy())
                print(syn_feature.data.cpu().numpy().shape, syn_label.data.cpu().numpy().shape)
                self.run_time1 = time.time() - self.begin_time

            print('unseen=%.4f, seen=%.4f, h=%.4f, best_u=%.4f, best_s=%.4f, best_h=%.4f, run_time=%.4f ' %
                  (cls.acc_unseen, cls.acc_seen, cls.H, self.best_u, self.best_s, self.best_h, self.run_time1))

        syn_feature, syn_label = self.generate_syn_feature(self.netG, self.data.unseenclasses, self.data.attribute,
                                                           self.syn_num)

        cls = classifier2.CLASSIFIER(syn_feature, self.map_label(syn_label, self.data.unseenclasses),
                                     self.data, self.data.unseenclasses.size(0), self.cuda, self.classifier_lr, 0.5,
                                     25, self.syn_num, False)
        if self.best_t < cls.acc:
            self.best_t = cls.acc
            self.run_time2 = time.time() - self.begin_time
        acc = cls.acc
        print('unseen class accuracy= %.4f, best_t=%.4f, run_time=%.4f '%(acc, self.best_t, self.run_time2))
        self.netG.train()

























































