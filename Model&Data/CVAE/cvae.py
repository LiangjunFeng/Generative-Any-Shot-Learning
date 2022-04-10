from keras.layers import Input, Dense, Lambda, merge, Dropout, BatchNormalization, concatenate
from keras.models import Model
import numpy as np
import keras.backend as K
import tensorflow.compat.v1 as tf
from scipy.io import loadmat
import os
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
import keras.backend.tensorflow_backend as KTF
from sklearn import svm
import time

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    per_class_accuracies = np.zeros(len(target_classes))
    test_label = test_label.squeeze()
    predicted_label = predicted_label.squeeze()
    for i in range(len(target_classes)):
        is_class = test_label == target_classes[i]
        per_class_accuracies[i] = ((predicted_label[is_class] == test_label[is_class]).sum()*1.0) / (is_class.sum()*1.0)
    return per_class_accuracies.mean()

def map_label(label, classes):
    mapped_label = np.zeros((label.shape[0],1), dtype = np.int)
    for i in range(len(classes)):
        mapped_label[label==classes[i]] = i
    return mapped_label


def process_few_shot_train(data, attsplits, num):
    labels = data["labels"]
    from copy import deepcopy
    copy_labels = deepcopy(labels).reshape(-1, 1)
    att = attsplits["att"]
    test_seen_loc = attsplits["test_seen_loc"]
    test_unseen_loc = attsplits["test_unseen_loc"]

    seen_classes = np.unique(np.ravel(labels)[test_seen_loc - 1]).tolist()
    copy_labels[test_seen_loc - 1] = -1
    add_seen_index = []
    for i in seen_classes:
        # print(np.where(copy_labels == i))
        add_seen_index += np.where(copy_labels == i)[0].tolist()[0:num]
    # print(add_seen_index)
    trainval_loc = np.array(add_seen_index).reshape(-1, 1) + 1
    print(trainval_loc.shape)
    if trainval_loc.shape[0] < 1024:
        n = int(1024 / trainval_loc.shape[0] + 1)
        trainval_loc = np.repeat(trainval_loc, n, axis=0)
    print(trainval_loc.shape)
    myLabel = {}
    myLabel["att"] = att
    myLabel["test_unseen_loc"] = test_unseen_loc
    myLabel["test_seen_loc"] = test_seen_loc
    myLabel["trainval_loc"] = trainval_loc
    return data, myLabel


def process_few_shot_test(data, attsplits, num):
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


def run(opt):
    begin_time = time.time()
    run_time1 = 0
    run_time2 = 0
    # ================== LAB RESOURCES ARE LIMITED=================== #

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    def get_session(gpu_fraction=0.4):
        '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        if num_threads:
            return tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        else:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    tf.compat.v1.keras.backend.set_session(get_session())
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    #===================================================================#
    # Some Constants

    m = opt.m             # batch_szie
    n_x = opt.n_x         # dim of feature embedding
    n_y = opt.n_y         # dim of label embedding
    n_z = opt.n_z         # dim of mu and sigma
    interNo = int(opt.interNo) # dim of H1
    n_epoch = opt.n_epoch # dim of H2
    path = opt.path       # path
    nSamples = opt.nSamples  # number of generated samples perclass
    # ============================================================ #

    input_ic = Input(shape=[n_x + n_y], name = 'img_class' )
    cond  = Input(shape=[n_y] , name='class')
    temp_h_q = Dense(interNo, activation='relu')(input_ic)
    h_q_zd = Dropout(rate=0.7)(temp_h_q)
    h_q = Dense(interNo, activation='relu')(h_q_zd)
    mu = Dense(n_z, activation='linear')(h_q)
    log_sigma = Dense(n_z, activation='linear')(h_q)

    def sample_z(args):
        mu, log_sigma = args
        eps = K.random_normal(shape=[n_z], mean=0., stddev=1.)
        return mu + opt.scale*K.exp(log_sigma / 2) * eps

    z = Lambda(sample_z)([mu, log_sigma])

    # Depending on the keras version...
    # z_cond = merge([z, cond] , mode='concat', concat_axis=1)
    z_cond = concatenate([z, cond])

    decoder_hidden = Dense(1024, activation='relu')
    decoder_out = Dense(n_x, activation='linear')
    h_p = decoder_hidden(z_cond)
    reconstr = decoder_out(h_p)
    vae = Model(inputs=[input_ic , cond], outputs=[reconstr])

    encoder = Model(inputs=[input_ic , cond], outputs=[mu])


    d_in = Input(shape=[n_z+n_y])
    d_h = decoder_hidden(d_in)
    d_out = decoder_out(d_h)
    decoder = Model(d_in, d_out)

    def vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        recon = K.sum(K.square(y_pred - y_true), axis=1)
        # recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
        #print 'kl : ' + str(kl)
        return K.mean(recon + opt.scale*kl)

    encoder.summary()
    decoder.summary()
    vae.compile(optimizer='adam', loss=vae_loss)

    # ======================================================================= #

    data = loadmat(path+opt.image_embedding+".mat")
    att_splits = loadmat(path + opt.class_embedding+"_splits.mat")

    if opt.num_shots > 0:
        if opt.few_train:
            data, att_splits = process_few_shot_train(data, att_splits, opt.num_shots)
        else:
            data, att_splits = process_few_shot_test(data, att_splits, opt.num_shots)

    features = data["features"].T
    labels = data["labels"]
    if labels.shape[0] == 1:
        labels = labels.T
    attributes = att_splits['att'].T

    trainData = features[att_splits['trainval_loc'].squeeze()-1]
    trainLabelVectors = attributes[(labels[att_splits['trainval_loc'].squeeze()-1]-1).squeeze()]

    X_train = np.concatenate([trainData , trainLabelVectors], axis=1)

    print('Fitting VAE Model...')
    print(X_train.shape, trainLabelVectors.shape)
    print(trainData.shape)
    vae.fit({'img_class' : X_train , 'class' : trainLabelVectors}, trainData, batch_size=m, nb_epoch=n_epoch)

    # =========================== UNSEEN CLASSES ======================================#

    testSeenData = features[att_splits['test_seen_loc'].squeeze()-1]
    testUnseenData = features[att_splits['test_unseen_loc'].squeeze()-1]

    trainLabels = labels[att_splits['trainval_loc'].squeeze()-1]-1
    testSeenLabel = labels[att_splits['test_seen_loc'].squeeze()-1]-1
    testUnseenLabel = labels[att_splits['test_unseen_loc'].squeeze()-1]-1

    testSeenClasses = np.unique(testSeenLabel).tolist()
    testUnseenClasses = np.unique(testUnseenLabel).tolist()

    print(testSeenClasses)
    print(testUnseenClasses)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # ==================================================

    pseudoTrainLabels =[]
    pseudoTrainAttr = []
    totalExs = len(testUnseenClasses)*nSamples

    with sess.as_default():
        noise_gen = K.random_normal(shape=(totalExs, n_z), mean=0., stddev=1.)

    for tc in testUnseenClasses:
        for ii in range(0,nSamples):
            pseudoTrainAttr.append(attributes[tc])
            pseudoTrainLabels.append(tc)
    # ===================================================

    pseudoTrainAttr = np.array(pseudoTrainAttr)
    pseudoTrainLabels = np.array(pseudoTrainLabels).reshape((-1,1))

    dec_ip = np.concatenate((noise_gen, pseudoTrainAttr), axis=1)
    pseudoTrainData = decoder.predict(dec_ip)


    #================================
    my_pseudoTrainLabels =[]
    my_pseudoTrainAttr = []
    my_totalExs = len(testUnseenClasses)*500

    with sess.as_default():
        my_noise_gen = K.random_normal(shape=(my_totalExs, n_z), mean=0., stddev=1.)

    for tc in testUnseenClasses:
        for ii in range(0,500):
            my_pseudoTrainAttr.append(attributes[tc])
            my_pseudoTrainLabels.append(tc)

    my_pseudoTrainAttr = np.array(my_pseudoTrainAttr)
    my_pseudoTrainLabels = np.array(my_pseudoTrainLabels)

    my_dec_ip = np.concatenate((my_noise_gen, my_pseudoTrainAttr), axis=1)
    my_pseudoTrainData = decoder.predict(my_dec_ip)

    syn_feature, syn_label = my_pseudoTrainData, my_pseudoTrainLabels
    np.save("./cvae_feat.npy", syn_feature)
    np.save("./cvae_label.npy", syn_label)
    print(syn_feature.shape, syn_label.shape)
    #================================


    ZSL_traindata = normalize(pseudoTrainData, axis=1)
    ZSL_testdata = normalize(testUnseenData, axis=1)
    ZSL_trainlabel = map_label(pseudoTrainLabels, testUnseenClasses)
    ZSL_testlabel = map_label(testUnseenLabel, testUnseenClasses)

    print(ZSL_traindata.shape, ZSL_trainlabel.shape)
    print(ZSL_testdata.shape, ZSL_testlabel.shape)

    print('Training SVM-100')
    clf5 = svm.SVC(C=100)
    # clf5  = GaussianNB()
    clf5.fit(ZSL_traindata, ZSL_trainlabel)
    print('Predicting...')
    pred = clf5.predict(ZSL_testdata)
    unseen_acc = compute_per_class_acc_gzsl(ZSL_testlabel, pred, list(range(len(testUnseenClasses))))
    run_time1 = time.time() - begin_time
    print("unseen acc: {0}, run_time={1}".format(unseen_acc, run_time1))


    if opt.generalized:

        pseudoSeenTrainLabels = []
        pseudoSeenTrainAttr = []
        totalExs = len(testSeenClasses) * nSamples

        with sess.as_default():
            noiseSeen_gen = K.random_normal(shape=(totalExs, n_z), mean=0., stddev=1.)

        for tc in testSeenClasses:
            for ii in range(0, nSamples):
                pseudoSeenTrainAttr.append(attributes[tc])
                pseudoSeenTrainLabels.append(tc)

        pseudoSeenTrainAttr = np.array(pseudoSeenTrainAttr)
        pseudoSeenTrainLabels = np.array(pseudoSeenTrainLabels).reshape((-1, 1))

        print(noiseSeen_gen.shape, pseudoSeenTrainAttr.shape)
        decSeen_ip = np.concatenate((noiseSeen_gen, pseudoSeenTrainAttr), axis=1)
        pseudoSeenTrainData = decoder.predict(decSeen_ip)

        GZSL_traindata = np.row_stack([pseudoSeenTrainData, pseudoTrainData])
        GZSL_trainlabel = np.row_stack([pseudoSeenTrainLabels, pseudoTrainLabels])
        GZSL_testdata = np.row_stack([testSeenData, testUnseenData])
        GZSL_testlabel = np.row_stack([testSeenLabel, testUnseenLabel])

        from scipy.io import savemat
        print(pseudoTrainData.shape, pseudoTrainLabels.shape, pseudoSeenTrainData.shape,
              pseudoSeenTrainLabels.shape, testUnseenData.shape, testUnseenLabel.shape, testSeenData.shape, testSeenLabel.shape)
        mydata = {"train_unseen_data": pseudoTrainData,
                  "train_unseen_label": pseudoTrainLabels,
                  "train_seen_data": pseudoSeenTrainData,
                  "train_seen_label": pseudoSeenTrainLabels,
                  "test_unseen_data": testUnseenData,
                  "test_unseen_label": testUnseenLabel,
                  "test_seen_data": testSeenData,
                  "test_seen_label": testSeenLabel}
        savemat("cvae_data.mat", mydata)
        print("cvae_data.mat is saved!")


        GZSL_traindata = normalize(GZSL_traindata, axis=1)
        GZSL_testdata = normalize(GZSL_testdata, axis=1)

        print(GZSL_traindata.shape, GZSL_trainlabel.shape)
        print(GZSL_testdata.shape, GZSL_testlabel.shape)

        print(np.unique(GZSL_trainlabel).tolist())

        print('Training SVM-100')
        clf5 = svm.SVC(C=100)
        # clf5  = GaussianNB()
        clf5.fit(GZSL_traindata, GZSL_trainlabel)
        print('Predicting...')
        pred = clf5.predict(GZSL_testdata)
        print(np.unique(pred).tolist())

        seen_acc = compute_per_class_acc_gzsl(GZSL_testlabel, pred, testSeenClasses)
        unseen_acc = compute_per_class_acc_gzsl(GZSL_testlabel, pred, testUnseenClasses)
        H = 2*seen_acc*unseen_acc/(seen_acc + unseen_acc)
        run_time2 = time.time() - begin_time
        print("seen acc: {0}, unseen acc: {1}, H acc: {2}, run_time={3}".format(seen_acc, unseen_acc, H, run_time2))









