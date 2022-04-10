from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util 
import classifier
import classifier2
import sys
import json
import model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import scipy.io as sio
import pickle
import time


def run(opt):
    print(opt)
    runing_parameters_logs = json.dumps(vars(opt), indent=4, separators=(',', ':'))

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if opt.dataset == "AWA1":
        #save_directory_path = "Ablation_Study_SR_loss_Weight : " + str(opt.correlation_penalty) + "Epsilon_Value : " + str(opt.epsilon)
        save_directory_path =  "GZSL_" + str(opt.generalized)  + "_" + "New_AWA1_" + "Unseen_loss_weight: " + str(opt.unseen_cls_weight)  + "_" + str(opt.dataset)  + "_" + str(opt.epsilon) + "_" + str(opt.correlation_penalty) + "_" + str(opt.unseen_start) + '_' + str(opt.CENT_LAMBDA)
    elif opt.generalized:
        save_directory_path =  "GZSL:" + str(opt.unseen_cls_weight) + "_" + str("out_test") + "_" + str(opt.dataset)  + "_" + str(opt.epsilon) + "_" + str(opt.correlation_penalty) + "_" + str(opt.unseen_start) + '_' + str(opt.CENT_LAMBDA)
    else:
        save_directory_path =  "ZSL" + "_" + str(opt.dataset)  + "_Epsilon : " + str(opt.epsilon) + "Upper_Epsilon : " + str(opt.upper_epsilon) + "_" + str(opt.correlation_penalty) + "_" + str(opt.unseen_start) + '_' + str(opt.CENT_LAMBDA)

    if not os.path.exists(save_directory_path):
        os.mkdir(save_directory_path)

    log_dir  = save_directory_path + '/log.txt'

    with open(log_dir, 'a') as f:
            f.write('Training Start:')
            f.write("Running Parameter Logs")
            f.write(runing_parameters_logs)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # load data
    data = util.DATA_LOADER(opt)
    print("# of training samples: ", data.ntrain)

    # initialize generator and discriminator
    netG = model.MLP_G(opt)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = model.MLP_CRITIC(opt)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    # classification loss, Equation (4) of the paper
    cls_criterion = nn.NLLLoss()

    input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
    input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
    noise = torch.FloatTensor(opt.batch_size, opt.nz)
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    input_label = torch.LongTensor(opt.batch_size)
    input_seen_map = torch.LongTensor(opt.batch_size)

    #Unseen data tensors
    input_att_u = torch.FloatTensor(opt.batch_size, opt.attSize)
    input_label_u = torch.LongTensor(opt.batch_size)
    input_unseen_map = torch.LongTensor(opt.batch_size)

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        input_res = input_res.cuda()
        noise, input_att = noise.cuda(), input_att.cuda()
        one = one.cuda()
        mone = mone.cuda()
        cls_criterion.cuda()
        input_label = input_label.cuda()
        data.tr_cls_centroid = torch.from_numpy(data.tr_cls_centroid).float().cuda()
        input_att_u = input_att_u.cuda()
        input_label_u = input_label_u.cuda()

    def sample():
        batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
        input_res.copy_(batch_feature)
        input_att.copy_(batch_att)
        input_seen_map.copy_(util.map_label(batch_label, data.seenclasses))
        input_label.copy_(batch_label)
        #util.map_label(batch_label, data.seenclasses))

    def unseen_sample():
        batch_label, batch_att = data.next_batch_unseen_class(opt.batch_size)
        input_label_u.copy_(batch_label)
        input_att_u.copy_(batch_att)
        input_unseen_map.copy_(util.map_label(batch_label, data.unseenclasses))

    def generate_syn_feature(netG, classes, attribute, num):
        nclass = classes.size(0)
        syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
        syn_label = torch.LongTensor(nclass*num)
        syn_att = torch.FloatTensor(num, opt.attSize)
        syn_noise = torch.FloatTensor(num, opt.nz)
        if opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()

        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            with torch.no_grad():
                output = netG(Variable(syn_noise), Variable(syn_att))

            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)

        return syn_feature, syn_label

    def centroid_calculation(syn_feature, syn_label):
        syn_feature = syn_feature.data.cpu().numpy()
        syn_label = syn_label.data.cpu().numpy()

        te_cls_centroid = np.zeros([data.ntest_class, syn_feature.shape[1]]).astype(np.float32)

        for i in range(data.ntest_class):
            te_cls_centroid[i] = np.mean(syn_feature[syn_label == i], axis=0)

        return te_cls_centroid

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def calc_gradient_penalty(netD, real_data, fake_data, input_att):
        #print real_data.size()
        alpha = torch.rand(opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if opt.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates, Variable(input_att))

        ones = torch.ones(disc_interpolates.size())
        if opt.cuda:
            ones = ones.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
        return gradient_penalty


    pretrain_cls = classifier.CLASSIFIER(data.train_feature, data.train_label, opt.nclass_all, opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

    sim_vector  =  np.zeros ((opt.nepoch, 10, 40))
    idex_vector =  np.zeros ((opt.nepoch , 10, 40))
    centroid_feat = np.zeros ((opt.nepoch, 10, data.train_feature.shape[1]))
    counter = 0

    main_seen_dic = {}
    main_unseen_dic = {}


    flag = False

    best_gzsl_acc = 0
    best_t = 0
    begin_time = time.time()
    run_time1 = 0
    run_time2 = 0
    for epoch in range(opt.nepoch):
        if epoch > 14 and flag == False and opt.dataset == "AWA1":
            opt.cls_weight = float(opt.cls_weight/2)
            flag = True

        FP = 0
        mean_lossD = 0
        mean_lossG = 0
        for i in range(0, data.ntrain, opt.batch_size):
            ############################
            # (1) Update D network: optimize WGAN-GP objective
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()
                # train with realG
                # sample a mini-batch
                sparse_real = opt.resSize - input_res[1].gt(0).sum()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                criticD_real = netD(input_resv, input_attv)
                criticD_real = criticD_real.mean()
                criticD_real.backward(mone)

                # train with fakeG
                noise.normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(noisev, input_attv)
                fake_norm = fake.data[0].norm()
                sparse_fake = fake.data[0].eq(0).sum()
                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(one)

                # gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
                gradient_penalty.backward()

                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()

            ############################
            # (2) Update G network: optimize WGAN-GP objective
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation

            netG.zero_grad()
            input_attv = Variable(input_att)
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            criticG_fake = netD(fake, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake
            # classification loss
            c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

            #Euclidian and Correlation Loss
            Euclidean_loss = Variable(torch.Tensor([0.0]), requires_grad=True).cuda()
            Correlation_loss = Variable(torch.Tensor([0.0]), requires_grad=True).cuda()

            for i in range(data.ntrain_class):
                sample_idx = (input_seen_map == i).data.nonzero().squeeze()
                if sample_idx.numel() == 0:
                    Euclidean_loss += 0.0
                else:
                    G_sample_cls = fake[sample_idx, :]
                    if sample_idx.numel() != 1:
                        generated_mean = G_sample_cls.mean(dim=0)
                    else:
                        generated_mean = G_sample_cls

                    Euclidean_loss += (generated_mean - data.tr_cls_centroid[i]).pow(2).sum().sqrt()
                    for n in range(opt.Neighbours):
                        Neighbor_correlation = cosine_similarity(generated_mean.data.cpu().numpy().reshape((1, generated_mean.shape[0])),
                                                data.tr_cls_centroid[data.idx_mat[i,n]].data.cpu().numpy().reshape((1, generated_mean.shape[0])))

                        lower_limit = data.semantic_similarity_seen [i,n] - opt.epsilon

                        if opt.dataset == "CUB1":
                            upper_limit = data.semantic_similarity_seen [i,n] + opt.upper_epsilon
                        else:
                            upper_limit = data.semantic_similarity_seen [i,n] + opt.epsilon

                        lower_limit = torch.as_tensor(lower_limit.astype('float'))
                        upper_limit = torch.as_tensor(upper_limit.astype('float'))
                        corr = torch.as_tensor(Neighbor_correlation[0][0].astype('float'))
                        margin = (torch.max(corr- corr, corr - upper_limit))**2 + (torch.max(corr- corr, lower_limit - corr ))**2
                        Correlation_loss += margin

            Euclidean_loss *= 1.0/data.ntrain_class * opt.CENT_LAMBDA
            Correlation_loss = (Correlation_loss/opt.Neighbours) * opt.correlation_penalty
            Correlation_loss.backward()

            errG = G_cost + opt.cls_weight*c_errG + Euclidean_loss
            errG.backward()
            optimizerG.step()

            if epoch > opt.unseen_start and opt.correlation_penalty > 0:
                ############################
                # (3) Update D network: optimize WGAN-GP objective, Unseen classes
                ###########################
                for p in pretrain_cls.model.parameters():
                    p.requires_grad = True

                for iter_d in range(1):
                    unseen_sample()
                    pretrain_cls.model.zero_grad()
                    # train with realG
                    # sample a mini-batch
                    input_attv_u = Variable(input_att_u)

                    # train with fakeG
                    noise.normal_(0, 1)
                    noisev = Variable(noise)
                    fake = netG(noisev, input_attv_u)

                    c_errG_zero = cls_criterion(pretrain_cls.model(fake), Variable(input_label_u))
                    c_errG_orginal = cls_criterion(pretrain_cls.model(input_res), Variable(input_label))

                    Correlation_loss_zero = Variable(torch.Tensor([0.0]), requires_grad= True).cuda() # My part :)
                    for i in range(data.ntest_class):
                        sample_idx = (input_unseen_map == i).data.nonzero().squeeze()
                        if sample_idx.numel() != 0:
                            G_sample_cls_zero = fake[sample_idx, :]
                            if sample_idx.numel() != 1:
                                generated_mean = G_sample_cls_zero.mean(dim=0)
                            else:
                                generated_mean = G_sample_cls_zero

                            for n in range(opt.Neighbours):
                                Neighbor_correlation = cosine_similarity(generated_mean.data.cpu().numpy().reshape((1, generated_mean.shape[0])),
                                                        data.tr_cls_centroid[data.unseen_idx_mat[i,n]].data.cpu().numpy().reshape((1, generated_mean.shape[0])))

                                lower_limit = data.semantic_similarity_unseen [i,n] - opt.epsilon

                                if opt.dataset == "CUB1":
                                    upper_limit = data.semantic_similarity_unseen [i,n] + opt.upper_epsilon
                                else:
                                    upper_limit = data.semantic_similarity_unseen [i,n] + opt.epsilon

                                lower_limit = torch.as_tensor(lower_limit.astype('float'))
                                upper_limit = torch.as_tensor(upper_limit.astype('float'))
                                corr = torch.as_tensor(Neighbor_correlation[0][0].astype('float'))
                                margin = (torch.max(corr- corr, corr - upper_limit))**2 + (torch.max(corr- corr, lower_limit - corr ))**2
                                Correlation_loss_zero += margin

                    Correlation_loss_zero = (Correlation_loss_zero/opt.Neighbours) * opt.correlation_penalty
                    Correlation_loss_zero.backward()

                    if opt.generalized:
                        errG_zero = opt.unseen_cls_weight*c_errG_zero  + opt.cls_weight*c_errG_orginal
                    else:
                        errG_zero = opt.unseen_cls_weight*c_errG_zero

                    errG_zero.backward()
                    optimizerG.step()
                    pretrain_cls.optimizer.step()

        mean_lossG /=  data.ntrain / opt.batch_size
        mean_lossD /=  data.ntrain / opt.batch_size

        log_text = 'This is for the seen training [%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f, E_dis : %.4f, Corr_Loss : %4f' % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG.item(), Euclidean_loss.item(), Correlation_loss.item())

        print(log_text)

        with open(log_dir, 'a') as f:
            f.write(log_text+'\n')

        if epoch > opt.unseen_start and opt.correlation_penalty > 0:
            if opt.generalized:
                log_text = 'This is for the unseen training [%d/%d] c_errG_zero:%.4f, Corr_Loss : %4f, Cls_Org :%4f' % (epoch, opt.nepoch, c_errG_zero.item(), Correlation_loss_zero.item(), c_errG_orginal.item())
            else:
                log_text = 'This is for the unseen training [%d/%d] c_errG_zero:%.4f, Corr_Loss : %4f' % (epoch, opt.nepoch, c_errG_zero.item(), Correlation_loss_zero.item())

            print(log_text)
            with open(log_dir, 'a') as f:
                f.write(log_text+'\n')

        # evaluate the model, set G to evaluation mode
        netG.eval()
        # Generalized zero-shot learning
        if opt.generalized:
            if opt.no_classifier == True:
                syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
                mapped_label = util.map_label(syn_label, data.unseenclasses)
                te_cls_centroid = centroid_calculation (syn_feature, mapped_label)
                counter = counter + 1
                acc_seen = pretrain_cls.val_gzsl(data.test_seen_feature, data.test_seen_label, data.test_seenclasses)
                acc_unseen = pretrain_cls.val_gzsl(data.test_unseen_feature, data.test_unseen_label, data.unseenclasses)
                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)

                if best_gzsl_acc <= H:
                    best_acc_seen, best_acc_unseen, best_gzsl_acc = acc_seen, acc_unseen, H

                    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, 500)
                    np.save("./lsrgan_feat.npy", syn_feature.data.cpu().numpy())
                    np.save("./lsrgan_label.npy", syn_label.data.cpu().numpy())
                    print(syn_feature.data.cpu().numpy().shape, syn_label.data.cpu().numpy().shape)

                    from scipy.io import savemat
                    print(syn_feature.cpu().detach().numpy().shape, syn_label.cpu().detach().numpy().shape,
                          data.train_feature.cpu().detach().numpy().shape,
                          data.train_label.cpu().detach().numpy().shape,
                          data.test_unseen_feature.cpu().detach().numpy().shape,
                          data.test_unseen_label.cpu().detach().numpy().shape,
                          data.test_seen_feature.cpu().detach().numpy().shape,
                          data.test_seen_label.cpu().detach().numpy().shape)
                    mydata = {"train_unseen_data": syn_feature.cpu().detach().numpy(),
                              "train_unseen_label": syn_label.cpu().detach().numpy(),
                              "train_seen_data": data.train_feature.cpu().detach().numpy(),
                              "train_seen_label": data.train_label.cpu().detach().numpy(),
                              "test_unseen_data": data.test_unseen_feature.cpu().detach().numpy(),
                              "test_unseen_label": data.test_unseen_label.cpu().detach().numpy(),
                              "test_seen_data": data.test_seen_feature.cpu().detach().numpy(),
                              "test_seen_label": data.test_seen_label.cpu().detach().numpy()}
                    savemat("lsrgan_data.mat", mydata)
                    print("lsrgan_data.mat is saved!")

                    run_time1 = time.time() - begin_time
                print('GZSL: seen=%.3f, unseen=%.3f, h=%.3f, best_seen=%.3f, best_unseen=%.3f, best_h=%.3f, run_time=%.4f'
                      % (acc_seen, acc_unseen, H, best_acc_seen, best_acc_unseen, best_gzsl_acc, run_time1))
            else:
                syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
                train_X = torch.cat((data.train_feature, syn_feature), 0)
                train_Y = torch.cat((data.train_label, syn_label), 0)
                nclass = opt.nclass_all
                cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)

                if best_t < cls.acc:
                    best_t = cls.acc
                    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, 500)
                    np.save("./lsrgan_feat.npy", syn_feature.data.cpu().numpy())
                    np.save("./lsrgan_label.npy", syn_label.data.cpu().numpy())
                    print(syn_feature.data.cpu().numpy().shape, syn_label.data.cpu().numpy().shape)
                    run_time2 = time.time() - begin_time
                acc = cls.acc
                print('unseen class accuracy= %.4f, best_t=%.4f, run_time=%.4f ' % (acc, best_t, run_time2))

        # Zero-shot learning
        if opt.dataset == "AWA1":
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            mapped_label = util.map_label(syn_label, data.unseenclasses)
            cls = classifier2.CLASSIFIER(syn_feature, mapped_label, data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
            if best_t < cls.acc:
                best_t = cls.acc
                syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, 500)
                np.save("./lsrgan_feat.npy", syn_feature.data.cpu().numpy())
                np.save("./lsrgan_label.npy", syn_label.data.cpu().numpy())
                print(syn_feature.data.cpu().numpy().shape, syn_label.data.cpu().numpy().shape)
                run_time2 = time.time() - begin_time
            acc = cls.acc
        else:
            acc = pretrain_cls.unesen_val(data.test_unseen_feature, data.test_unseen_label, data.unseenclasses)
            if best_t < acc:
                best_t = acc
                syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, 500)
                np.save("./lsrgan_feat.npy", syn_feature.data.cpu().numpy())
                np.save("./lsrgan_label.npy", syn_label.data.cpu().numpy())
                print(syn_feature.data.cpu().numpy().shape, syn_label.data.cpu().numpy().shape)
                run_time2 = time.time() - begin_time
        print('unseen class accuracy= %.4f, best_t=%.4f, run_time=%.4f ' % (acc, best_t, run_time2))

        # reset G to training mode
        netG.train()