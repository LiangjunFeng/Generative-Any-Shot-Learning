#author: akshitac8
from __future__ import print_function
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import random
# load files
import networks.TFVAEGAN_model as model
import datasets.action_util as util
import classifiers.classifier_actions as classifier
import classifiers.classifier_entropy as classifier_entropy
from config_actions import opt

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
data = util.DATA_LOADER(opt)
print("Training samples: ", data.ntrain)
print("Dataset: ", opt.dataset)

# Init modules: Encoder, Generator, Discriminator
netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
# Init models: Feedback module, auxillary module
netF = model.Feedback(opt)
netDec = model.AttDec(opt,opt.attSize)

print(netE)
print(netG)
print(netD)

print(netF)
print(netDec)

# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_bce_att = torch.FloatTensor(opt.batch_size, opt.attSize)
one = torch.FloatTensor([1])
mone = one * -1
##########
# Cuda
if opt.cuda:
    netG.cuda()
    netD.cuda()
    netE.cuda()
    netDec.cuda()
    netF.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    input_bce_att = input_bce_att.cuda()
    one = one.cuda()
    mone = mone.cuda()

def loss_fn(recon_x, x, mean, log_var):
    #vae loss L_bce + L_kl
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(), size_average=False)
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)

def WeightedL1(pred, gt, bce=False, gt_bce=None):
    #semantic embedding cycle-consistency loss
    if bce:
        BCE = torch.nn.functional.binary_cross_entropy(pred+1e-12, gt_bce.detach(),size_average=False)
        return BCE.sum()/pred.size(0)
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)
           
def feedback_module(gen_out, att, netG, netDec, netF):
    syn_fake = netG(gen_out, c=att)
    #recons = netDec(syn_fake)
    recons_hidden_feat = netDec(syn_fake).getLayersOutDet()
    feedback_out = netF(recons_hidden_feat)
    syn_fake = netG(gen_out, a1=opt.a1, c=att, feedback_layers=feedback_out)
    return syn_fake

def sample():
    #data loader
    batch_feature, batch_att, batch_bce_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_()    
    input_bce_att.copy_(batch_bce_att, batch_att)

def generate_syn_feature(netG, classes, attribute, num, netF=None, netDec=None):
    #unseen feature synthesis
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
        #replicate the attributes
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise,volatile=True)
        syn_attv = Variable(syn_att,volatile=True)
        output = feedback_module(gen_out=syn_noisev, att=syn_attv, netG=netG, netDec=netDec, netF=netF)
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)
    return syn_feature, syn_label

#setup optimizer
optimizerD         = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE         = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerG         = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF         = optim.Adam(netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec       = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD,real_data, fake_data, input_att):
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

best_zsl_acc = 0
if opt.gzsl:
    best_gzsl_acc = 0

#Training loop
for epoch in range(0,opt.nepoch):
    #feedback training loop
    for loop in range(0,opt.feedback_loop):
        for i in range(0, data.ntrain, opt.batch_size):
            #########Discriminator training ##############

            #unfreeze discrimator
            for p in netD.parameters():
                p.requires_grad = True

            #unfreeze deocder
            for p in netDec.parameters():
                p.requires_grad = True

            # Train D1 and Decoder
            gp_sum = 0
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()          
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                
                #Training the auxillary module
                netDec.zero_grad()
                recons = netDec(input_resv)
                R_cost = opt.recons_weight*WeightedL1(recons, input_attv, bce=opt.bce_att, gt_bce=Variable(input_bce_att)) 
                R_cost.backward()
                optimizerDec.step()
                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD*criticD_real.mean()
                criticD_real.backward(mone)
                if opt.encoded_noise:        
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size])
                    if opt.cuda: eps = eps.cuda()
                    eps = Variable(eps)
                    latent_code = eps * std + means
                else:
                    noise.normal_(0, 1)
                    latent_code = Variable(noise)

                #feedback loop
                if loop == 1:
                    fake = feedback_module(gen_out=latent_code, att=input_attv, netG=netG, netDec=netDec, netF=netF)
                else:
                    fake = netG(latent_code, c=input_attv)
                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD*criticD_fake.mean()
                criticD_fake.backward(one)
                # gradient penalty
                gradient_penalty = opt.gammaD*calc_gradient_penalty(netD, input_res, fake.data, input_att)
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()         
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty #add Y here and #add vae reconstruction loss
                optimizerD.step()

            # Adaptive lambda
            gp_sum /= (opt.gammaD*opt.lambda1*opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            #############netG training ##############
            # Train netG and Decoder
            for p in netD.parameters():
                p.requires_grad = False
            
            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netDec.parameters():
                    p.requires_grad = False
                        
            netE.zero_grad()
            netG.zero_grad()
            netF.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            #This is outside the opt.encoded_noise condition because of the vae loss
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size])
            if opt.cuda: eps = eps.cuda()
            eps = Variable(eps)
            latent_code = eps * std + means
            if loop == 1:
                recon_x = feedback_module(gen_out=latent_code, att=input_attv, netG=netG, netDec=netDec, netF=netF)
            else:
                recon_x = netG(latent_code, c=input_attv)

            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var)
            errG = vae_loss_seen

            if opt.encoded_noise:
                criticG_fake = netD(recon_x,input_attv).mean()
                fake = recon_x 
            else:
                noise.normal_(0, 1)
                latent_code_noise = Variable(noise)
                if loop == 1:
                    fake = feedback_module(gen_out=latent_code_noise, att=input_attv, netG=netG, netDec=netDec, netF=netF)
                else:
                    fake = netG(latent_code_noise, c=input_attv)
                criticG_fake = netD(fake,input_attv).mean()    
            
            G_cost = -criticG_fake
            # Add vae loss and generator loss
            errG += opt.gammaG*G_cost
            netDec.zero_grad()
            recons_fake = netDec(fake)
            R_cost = WeightedL1(recons_fake, input_attv, bce=opt.bce_att, gt_bce=Variable(input_bce_att))
            # Add reconstruction loss
            errG += opt.recons_weight * R_cost
            errG.backward()
            optimizerE.step()
            optimizerG.step()
            if loop == 1:
                optimizerF.step()
            if opt.recons_weight > 0 and not opt.freeze_dec: # not train decoder at feedback time
                optimizerDec.step()        
    # Print losses
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f'% \
    (epoch, opt.nepoch, D_cost.data[0], G_cost.data[0], Wasserstein_D.data[0],vae_loss_seen.data[0]),end=" ")
    # Evaluation
    netG.eval()
    netDec.eval()
    netF.eval()
    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num, netF=netF, netDec=netDec)
    # Generalized zero-shot learning
    if opt.gzsl_od:
        # OD based GZSL
        seen_class = data.seenclasses.size(0)
        clsu = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), \
                                    opt.cuda, _nepoch=25, _batch_size=opt.syn_num, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
        clss = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label,data.seenclasses), data, seen_class, opt.cuda, \
                                    _nepoch=25, _batch_size=opt.syn_num, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
        clsg = classifier_entropy.CLASSIFIER(data.train_feature, util.map_label(data.train_label,data.seenclasses), data, seen_class, \
                                            syn_feature, syn_label, opt.cuda, clss, clsu, _batch_size=128, \
                                            netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
        if best_gzsl_acc < clsg.H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = clsg.acc_seen, clsg.acc_unseen, clsg.H
        print('GZSL-OD: Acc seen=%.4f, Acc unseen=%.4f, h=%.4f' % (clsg.acc_seen, clsg.acc_unseen, clsg.H))

    # Zero-shot learning
    # Train ZSL classifier
    zsl_cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), \
                                    opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, generalized=False, netDec=netDec, \
                                    dec_size=opt.attSize, dec_hidden_size=4096)
    acc = zsl_cls.acc
    if best_zsl_acc < acc:
        best_zsl_acc = acc
    print('ZSL: unseen accuracy=%.4f' % (acc))
    # reset modules to training mode
    netG.train()
    netDec.train()
    netF.train()

#Best results
print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)
if opt.gzsl_od:
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)
    
