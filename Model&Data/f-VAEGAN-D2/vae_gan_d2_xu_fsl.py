from __future__ import print_function
import argparse
import os
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import util_fewshot
import classifier_fsl
import model
import time
import numpy as np


def run(opt):
    opt.nz = opt.latent_size
    print(opt)


    logger = util_fewshot.Logger(opt.outname)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    logger.write('Random Seed=%d\n' % (opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # load data
    data = util_fewshot.DATA_LOADER(opt)
    print("# of training samples: ", data.ntrain)

    # initialize generator and discriminator
    netG = model.Decoder(opt.decoder_layer_sizes, opt.latent_size, opt.attSize)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = model.MLP_CRITIC(opt)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    Encoder = model.Encoder(opt.encoder_layer_sizes, opt.latent_size, opt.attSize)
    if opt.Encoder != '':
        Encoder.load_state_dict(torch.load(opt.Encoder))
    print(Encoder)

    input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
    input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
    noise = torch.FloatTensor(opt.batch_size, opt.nz)
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        Encoder.cuda()
        noise = noise.cuda()
        input_res = input_res.cuda()
        input_att = input_att.cuda()
        one = one.cuda()
        mone = mone.cuda()

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return BCE + KLD


    def sample():
        batch_feature, batch_att = data.next_seen_batch(opt.batch_size)
        input_res.copy_(batch_feature)
        input_att.copy_(batch_att)


    def generate_syn_feature(vae, classes, attribute, num):
        nclass = classes.size(0)
        syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
        syn_label = torch.LongTensor(nclass*num)
        syn_att = torch.FloatTensor(num, opt.attSize)
        syn_noise = torch.FloatTensor(num, opt.nz)
        if opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()

        with torch.no_grad():
            for i in range(nclass):
                iclass = classes[i]
                iclass_att = attribute[iclass]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)
                output = netG(syn_noise, syn_att)
                syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                syn_label.narrow(0, i*num, num).fill_(iclass)

        return syn_feature, syn_label

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerE = optim.Adam(Encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def calc_gradient_penalty(netD, real_data, fake_data, input_att):
        #print real_data.size()
        alpha = torch.rand(opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if opt.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = interpolates.requires_grad_(True)
        disc_interpolates = netD(interpolates, input_att)

        ones = torch.ones(disc_interpolates.size())
        if opt.cuda:
            ones = ones.cuda()
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
        return gradient_penalty

    best_acc = 0
    best_s = 0
    best_u = 0
    best_h = 0
    best_t = 0

    begin_time = time.time()
    run_time1 = 0
    run_time2 = 0
    for epoch in range(opt.nepoch):
        FP = 0
        mean_lossD = 0
        mean_lossG = 0
        for i in range(0, data.ntrain, opt.batch_size):
            ############################
            # (1) Update D network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            for iter_d in range(opt.critic_iter):
                # sample a mini-batch
                sample()
                netD.zero_grad()
                # train with realG
                criticD_real = netD(input_res, input_att)
                criticD_real = criticD_real.mean()
                criticD_real.backward(mone)

                # non-conditional D on unpaired real data

                # train with fakeG
                noise.normal_(0, 1)
                fake = netG(noise, input_att)
                criticD_fake = netD(fake.detach(), input_att)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(one)

                # gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
                gradient_penalty.backward()


                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()
            ############################
            # (2) Update G network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation

            netG.zero_grad()
            Encoder.zero_grad()
            # netG latent code vae loss
            mean, log_var = Encoder(input_res, input_att)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cuda()
            z = eps * std + mean
            recon_x = netG(z, input_att)
            vae_loss = loss_fn(recon_x, input_res, mean, log_var)
            # netG latent code gan loss
            criticG_fake = netD(recon_x, input_att)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake
            # net G fake data
            fake_v = netG(noise, input_att)
            criticG_fake2 = netD(fake_v, input_att)
            criticG_fake2 = criticG_fake2.mean()
            G_cost += -criticG_fake2


            loss = opt.gan_weight * G_cost + opt.vae_weight * vae_loss
            loss.backward()
            optimizerG.step()
            optimizerE.step()

        print('[%d/%d] Wasserstein_dist: %.4f, vae_loss:%.4f'
                  % (epoch, opt.nepoch, Wasserstein_D.data.item(), vae_loss.data.item()))
        logger.write('[%d/%d] Wasserstein_dist: %.4f, vae_loss:%.4f\n'
                  % (epoch, opt.nepoch, Wasserstein_D.data.item(), vae_loss.data.item()))

        # evaluate the model, set G to evaluation mode
        netG.eval()
        # Generalized few-shot learning
        if opt.generalized:
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)
            cls = classifier_fsl.CLASSIFIER(train_X, train_Y, data, data.ntrain_class, opt.cuda, opt.classifier_lr, 0.5, opt.nepoch_classifier, opt.syn_num, True)

            if best_h < cls.acc_all:
                best_h = cls.acc_all
                best_u = cls.acc_novel
                best_s = cls.acc_base
                syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, 500)
                np.save("./fvaegan_feat.npy", syn_feature.data.cpu().numpy())
                np.save("./fvaegan_label.npy", syn_label.data.cpu().numpy())
                print(syn_feature.data.cpu().numpy().shape, syn_label.data.cpu().numpy().shape)
                run_time1 = time.time() - begin_time

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
                savemat("fvaegan_data.mat", mydata)
                print("fvaegan_data.mat is saved!")

            print('unseen=%.4f, seen=%.4f, h=%.4f, best_u=%.4f, best_s=%.4f, best_h=%.4f, run_time=%.4f' %
                  (cls.acc_novel, cls.acc_base, cls.acc_all, best_u, best_s, best_h, run_time1))
            logger.write('acc_H=%.4f, acc_seen=%.4f, acc_unseen=%.4f\n' % (cls.acc_all, cls.acc_base, cls.acc_novel))
            acc = cls.acc_all
        # Few-shot learning
        # else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier_fsl.CLASSIFIER(syn_feature, util_fewshot.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, opt.nepoch_classifier, opt.syn_num, False)

        if best_t < cls.acc:
            best_t = cls.acc
            run_time2 = time.time() - begin_time
        acc = cls.acc
        print('unseen class accuracy= %.4f, best_t=%.4f, run_time=%.4f'%(acc, best_t, run_time2))
        logger.write('novel class accuracy= %.4f\n' % acc)

        # reset G to training mode
        netG.train()
