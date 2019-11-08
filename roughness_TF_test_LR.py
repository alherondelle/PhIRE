import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import strftime, time
import sys
sys.path.append('../utils')
from utils import *
from srgan_rough import SRGAN

# Suppress TensorFlow warnings about not using GPUs because they're annoying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Needed this line to run TensorFlow for some reason
# Would prefer to find another workaround if possible
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Network training meta-parameters
learning_rate = 1e-4 # Learning rate for gradient descent (may decrease to 1e-5 after initial training)
batch_size = 100 # Batch size for training (decrease this if you get memory errors, particularly on GPUs)
N_epochs = 1000 # Number of epochs of training
epoch_shift = 0 # If reloading previously trained network, what epoch to start at
save_every = 1 # How frequently (in epochs) to save model weights
print_every = 1000 # How frequently (in iterations) to write out performance

# Set loss function for training
# loss_type options:
#         MSE: pixel-wise mean squared error of output (no perceptual losses)
#          AE: perceptual loss using physics-informed autoencoder
#    MSEandAE: combination of MSE and AE
#         VGG: perceptual loss using VGG-19 image classification network
#   MSEandVGG: combination of MSE and VGG
loss_type = 'MSE'
if loss_type in ['AE', 'MSEandAE']:
    loss_model = '../autoencoder/model/latest2d/autoencoder'
    #loss_model = '../autoencoder/model/latest3d/autoencoder'
elif loss_type in ['VGG', 'MSEandVGG']:
    loss_model = '../VGG19/model/vgg19'

# Set various paths for where to save data
data_type = 'ua-va_wtk'
trainOn_testOn = "_10_"+ data_type +"_us_2c"
now = strftime('%Y%m%d-%H%M%S') + trainOn_testOn
model_name = '/'.join(['model', now])
img_path = '/'.join(['training_imgs', now])
layerviz_path = '/'.join(['layer_viz_imgs', now])
log_path ='/'.join(['training_logs', now])
print("model name: ", model_name)
#test_data_path ='/'.join(['..', 'data', now])
path_prefix = "../../../../projects/dles/SR_data/WIND/"
test_data_path = path_prefix + 'test_data' + now + "/"

#surface roughness
wtk_roughness_path = path_prefix + 'WTK_roughness_MR_normed.npy'
#ccsm_roughness_path = path_prefix + 'CCSM_roughness_MR.npy'
# Paths for data to load
train_path = path_prefix + "wtk_4hrRand_us_2007-2013_ua-va_slices_train_LR_10_MR_100_roughness.tfrecord"
test_path = path_prefix + "wtk_4hrRand_us_2007-2013_ua-va_slices_test_LR_10_MR_100_roughness.tfrecord"
ccsm_test_path = ""

def roughnessGrabber(lat, lon, width, height,isWTK = True):
    #WIDTH AND HEIGHT SHOULD BE SIZE OF IMAGE SO THAT THE SLICES MATCH.
    #print('roughnessGrabber: ')
    roughness = None
    roughness_array = None
    lat_coords, lon_coords = None, None
    if isWTK:
        roughness = np.load(wtk_roughness_path)
    else:
        roughness = np.load(ccsm_roughness_path)
    for i in range(lat.shape[0]):
        lat_coords = np.where(roughness[:,:,0] == lat[i,:])
        lon_coords = np.where(roughness[:,:,1] == lon[i,:])

        #print(lat_coords, lon_coords)
        r = roughness[lat_coords[0][0]:lat_coords[0][0]+width, lon_coords[1][0]:lon_coords[1][0]+height, 2]
        #print(lat_coords[0][0], lat_coords[0][0]+width, roughness[:,:,2].shape)
        if roughness_array is None:
            roughness_array = np.reshape(r, (1, r.shape[0], r.shape[1], 1))
        else:
            roughness_array = np.concatenate((roughness_array, np.reshape(r, (1, r.shape[0], r.shape[1], 1))), axis = 0)
        #print(roughness_array.shape, np.reshape(r, (1, r.shape[0], r.shape[1], 1)))
    #return roughness[:width, :height,2]
    return roughness_array

def pre_train(mu_sig):
    """Pretrain network (i.e., no adversarial component)."""

    print('Initializing network ...', mu_sig, end=' ')
    # Set high- and low-res data place holders. Make sure the sizes match the data
    x_LR = tf.placeholder(tf.float32, [None,  10,  10, 2])
    x_HR = tf.placeholder(tf.float32, [None, 100, 100, 2])
    x_rough = tf.placeholder(tf.float32, [None, 100, 100,1])
    # Set super resolution scaling. Product of terms in r must match scaling from low-res to high-res.
    r = [2, 5]

    # Initialize network and set optimizer
    model = SRGAN(x_LR, x_HR, x_rough, r=r, status='pre-training', loss_type=loss_type)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    g_train_op = optimizer.minimize(model.g_loss, var_list=model.g_variables)
    init = tf.global_variables_initializer()
    g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
    print('Done.')

    print('Number of generator params: %d' %(count_params(model.g_variables)))

    print('Building data pipeline ...', end=' ')
    ds_train = tf.data.TFRecordDataset(train_path)
    ds_train = ds_train.map(lambda xx: _parse_WTK_(xx, mu_sig)).shuffle(1000).batch(batch_size)

    ds_test = tf.data.TFRecordDataset(test_path)
    ds_test = ds_test.map(lambda xx: _parse_WTK_(xx, mu_sig)).batch(batch_size)

    iterator = tf.data.Iterator.from_structure(ds_train.output_types,
                                               ds_train.output_shapes)
    idx, LR_out, HR_out, lat, lon = iterator.get_next()
    #print(roughnessGrabber(lat, lon, HR_out.shape[1], HR_out.shape[2]).shape)

    init_iter_train = iterator.make_initializer(ds_train)
    init_iter_test  = iterator.make_initializer(ds_test)
    print('Done.')

    # Create summary values for TensorBoard
    g_loss_iters_ph = tf.placeholder(tf.float32, shape=None)
    g_loss_iters = tf.summary.scalar('g_loss_vs_iters', g_loss_iters_ph)
    iter_summ = tf.summary.merge([g_loss_iters])

    g_loss_epoch_ph = tf.placeholder(tf.float32, shape=None)
    g_loss_epoch = tf.summary.scalar('g_loss_vs_epochs', g_loss_epoch_ph)
    epoch_summ = tf.summary.merge([g_loss_epoch])

    summ_train_writer = tf.summary.FileWriter(log_path+'-train', tf.get_default_graph())
    summ_test_writer  = tf.summary.FileWriter(log_path+'-test',  tf.get_default_graph())


    with tf.Session() as sess:
        print('Training network ...')

        sess.run(init)

        # Load previously trained network
        #g_saver.restore(sess, 'model/pretrain/SRGAN')
        if epoch_shift > 0:
            print('Loading pre-trained network...', end=' ')
            g_saver.restore(sess, 'model/20191016-095834_10_ua-va_wtk_us_2c/pretrain00203/SRGAN_pretrain')
            print('Done.')

        # Load perceptual loss network data if necessary
        if loss_type in ['AE', 'MSEandAE', 'VGG', 'MSEandVGG']:
            # Restore perceptual loss network, if necessary
            print('Loading perceptual loss network...', end=' ')
            var = tf.global_variables()
            if loss_type in ['AE', 'MSEandAE']:
                loss_var = [var_ for var_ in var if 'encoder' in var_.name]
            elif loss_type in ['VGG', 'MSEandVGG']:
                loss_var = [var_ for var_ in var if 'vgg19' in var_.name]
            saver = tf.train.Saver(var_list=loss_var)
            saver.restore(sess, loss_model)
            print('Done.')

        # Start training
        iters = 0
        for epoch in range(epoch_shift+1, epoch_shift+N_epochs+1):
            print('Epoch: '+str(epoch))
            start_time = time()

            # Loop through training data
            sess.run(init_iter_train)
            try:
                epoch_g_loss, epoch_d_loss, N_train = 0, 0, 0
                while True:
                    iters += 1

                    batch_idx, batch_LR, batch_HR, batch_lat, batch_lon = sess.run([idx, LR_out, HR_out, lat, lon])
                    #print("HR        ", np.min(batch_HR), np.mean(batch_HR), np.max(batch_HR))
                    batch_rough = roughnessGrabber(batch_lat, batch_lon, batch_HR.shape[1], batch_HR.shape[2], True)
                    #print("roughness ", np.min(batch_rough), np.mean(batch_rough), np.max(batch_rough))
                    #batch_rough = np.reshape(batch_rough, (batch_rough.shape[0], batch_rough.shape[1], batch_rough.shape[2], 1))
                    #print("shape ", batch_rough.shape)
                    '''nan_arr = np.zeros((batch_HR.shape[1], 10), dtype=np.float32)
                    nan_arr.fill(np.nan)

                    for i, b_idx in enumerate(batch_idx):
                        idx_path = '/'.join(['test', 'surfaceRoughness', 'img{0:04d}'.format(b_idx)])
                        print(idx_path)
                        if not os.path.exists(idx_path):
                            os.makedirs(idx_path)
                        plt.figure(figsize = (10,10))
                        plt.subplot(121)
                        plt.imshow(batch_HR[i, :, :, 0], cmap='jet', origin='lower')
                        plt.title("ua")
                        plt.subplot(122)
                        plt.imshow(batch_rough[i, :, :, 0], cmap='Greys', origin='lower')
                        plt.title("surface")

                        plt.savefig('/'.join([idx_path, 'surfaceRoughness_test_2.png']), bbox_inches = 'tight' )'''

                    feed_dict = {x_HR:batch_HR, x_LR:batch_LR, x_rough:batch_rough}
                    batch_SR = sess.run(model.x_SR, feed_dict=feed_dict) #try normalizing Surface.
                    print(np.min(batch_SR), np.mean(batch_SR), np.max(batch_SR), batch_SR.shape) #print out for all channels
                    #print(np.min(batch_SR), np.mean(batch_SR), np.max(batch_SR))
                    sess.run(g_train_op, feed_dict=feed_dict)
                    gl = sess.run(model.g_loss, feed_dict=feed_dict)

                    summ = sess.run(iter_summ, feed_dict={g_loss_iters_ph: gl})
                    summ_train_writer.add_summary(summ, iters)
                    N_batch = batch_SR.shape[0]
                    epoch_g_loss += gl*N_batch
                    N_train += N_batch

                    if (iters % print_every) == 0:
                        print('Iterations=%d, G loss=%.5f' %(iters, gl))

            except tf.errors.OutOfRangeError:
                if (epoch % save_every) == 0:
                    model_epoch = '/'.join([model_name, 'pretrain{0:05d}'.format(epoch)])
                    if not os.path.exists(model_epoch):
                        os.makedirs(model_epoch)
                    g_saver.save(sess, '/'.join([model_epoch, 'SRGAN_pretrain']))

                g_loss_train = epoch_g_loss/N_train

            # Loop through test data
            sess.run(init_iter_test)
            try:
                test_out = None
                epoch_g_loss, N_test = 0, 0
                while True:
                    batch_idx, batch_LR, batch_HR, batch_lat, batch_lon = sess.run([idx, LR_out, HR_out, lat, lon])

                    batch_rough = roughnessGrabber(batch_lat, batch_lon, batch_HR.shape[1], batch_HR.shape[2], True)
                    N_batch = batch_LR.shape[0]

                    feed_dict = {x_HR:batch_HR, x_LR:batch_LR, x_rough:batch_rough}
                    batch_SR, gl = sess.run([model.x_SR, model.g_loss], feed_dict=feed_dict)

                    epoch_g_loss += gl*N_batch
                    N_test += N_batch
                    #print("line 180, ", type(mu_sig[1]), type(mu_sig[0]), np.amin(batch_LR), np.mean(batch_LR), np.amax(batch_LR))

                    batch_LR = mu_sig[1]*batch_LR + mu_sig[0]
                    batch_SR = mu_sig[1]*batch_SR + mu_sig[0]
                    batch_HR = mu_sig[1]*batch_HR + mu_sig[0]
                    if (epoch % save_every) == 0:
                        nan_arr = np.zeros((batch_HR.shape[1], 10), dtype=np.float32)
                        nan_arr.fill(np.nan)
                        nan_arr2 = np.zeros((10, 3*batch_HR.shape[2] + 20), dtype=np.float32)
                        nan_arr2.fill(np.nan)
                        #print(batch_rough[i,:,:,0].shape)
                        #for i, idx in enumerate(batch_LR.shape[0]): # NEED TO INDEX ALL THE DATA
                        for i, b_idx in enumerate(batch_idx):
                            print(batch_rough[i,:,:,0].shape)
                            idx_path = '/'.join([img_path, 'pretrain', 'img{0:04d}'.format(b_idx)])
                            if not os.path.exists(idx_path):
                                os.makedirs(idx_path)
                            ua_img = np.concatenate((np.repeat(np.repeat(batch_LR[i, :, :, 0], np.prod(r), axis=0), np.prod(r), axis=1), \
                                                                          nan_arr, \
                                                                          batch_SR[i, :, :, 0], \
                                                                          nan_arr, \
                                                                          batch_HR[i, :, :, 0]), axis=1)
                            va_img = np.concatenate((np.repeat(np.repeat(batch_LR[i, :, :, 1], np.prod(r), axis=0), np.prod(r), axis=1), \
                                                                          nan_arr, \
                                                                          batch_SR[i, :, :, 1], \
                                                                          nan_arr, \
                                                                          batch_HR[i, :, :, 1]), axis=1)
                            cat_img = np.concatenate((ua_img, nan_arr2, va_img), axis=0)
                            plt.imsave('/'.join([idx_path, 'epoch{0:05d}'.format(epoch)+'.png']), cat_img, cmap='jet', format='png', origin='lower')
                            if epoch = epoch_shift+1:
                                plt.imsave('/'.join([idx_path, 'epoch{0:05d}'.format(epoch)+'surface.png']), batch_rough[i,:,:,0], cmap='Greys', format='png', origin='lower')
                            if test_out is None:
                                test_out = np.expand_dims(batch_SR[i], 0)
                            else:
                                test_out = np.concatenate((test_out, np.expand_dims(batch_SR[i], 0)), axis=0)

            except tf.errors.OutOfRangeError:
                g_loss_test = epoch_g_loss/N_test

                if not os.path.exists(test_data_path):
                    os.makedirs(test_data_path)
                if (epoch % save_every) == 0:
                    np.save(test_data_path +'/WTK_test_SR_epoch{0:05d}'.format(epoch)+'.npy', test_out)

                # Write performance to TensorBoard
                summ = sess.run(epoch_summ, feed_dict={g_loss_epoch_ph: g_loss_train})
                summ_train_writer.add_summary(summ, epoch)

                summ = sess.run(epoch_summ, feed_dict={g_loss_epoch_ph: g_loss_test})
                summ_test_writer.add_summary(summ, epoch)

                print('Epoch took %.2f seconds\n' %(time() - start_time))

        # Save model after training is completed
        if not os.path.exists(model_name):
            os.makedirs(model_name)
        g_saver.save(sess, '/'.join([model_name, 'pretrain', 'SRGAN_pretrain']))


def train(mu_sig):
    """Train network using GANs. Only run this after model has been sufficiently pre-trained."""
    save_every = 1
    print('Initializing network ...', end=' ')
    tf.reset_default_graph()

    # Set high- and low-res data place holders. Make sure the sizes match the data
    x_LR = tf.placeholder(tf.float32, [None,  10,  10, 2])
    x_HR = tf.placeholder(tf.float32, [None, 100, 100, 2])
    x_rough = tf.placeholder(tf.float32, [None, 100, 100,1])

    # Set super resolution scaling. Product of terms in r must match scaling from low-res to high-res.
    r = [2, 5]

    # Initialize network and set optimizer
    model = SRGAN(x_LR, x_HR, r=r, status='training', loss_type=loss_type)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    g_train_op = optimizer.minimize(model.g_loss, var_list=model.g_variables)
    d_train_op = optimizer.minimize(model.d_loss, var_list=model.d_variables)
    init = tf.global_variables_initializer()
    g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
    gd_saver = tf.train.Saver(var_list=(model.g_variables+model.d_variables), max_to_keep=10000)
    print('Done.')

    print('Number of generator params: %d' %(count_params(model.g_variables)))
    print('Number of discriminator params: %d' %(count_params(model.d_variables)))

    print('Building data pipeline ...', end=' ')
    ds_train = tf.data.TFRecordDataset(train_path)
    ds_train = ds_train.map(lambda xx: _parse_WTK_(xx, mu_sig)).shuffle(1000).batch(batch_size)

    ds_test = tf.data.TFRecordDataset(test_path)
    ds_test = ds_test.map(lambda xx: _parse_WTK_(xx, mu_sig)).batch(batch_size)

    iterator = tf.data.Iterator.from_structure(ds_train.output_types,
                                               ds_train.output_shapes)
    idx, LR_out, HR_out, lat, lon = iterator.get_next()

    init_iter_train = iterator.make_initializer(ds_train)
    init_iter_test  = iterator.make_initializer(ds_test)
    print('Done.')

    # Create summary values for TensorBoard
    g_loss_iters_ph = tf.placeholder(tf.float32, shape=None)
    d_loss_iters_ph = tf.placeholder(tf.float32, shape=None)
    g_loss_iters = tf.summary.scalar('g_loss_vs_iters', g_loss_iters_ph)
    d_loss_iters = tf.summary.scalar('d_loss_vs_iters', d_loss_iters_ph)
    iter_summ = tf.summary.merge([g_loss_iters, d_loss_iters])

    g_loss_epoch_ph = tf.placeholder(tf.float32, shape=None)
    d_loss_epoch_ph = tf.placeholder(tf.float32, shape=None)
    g_loss_epoch = tf.summary.scalar('g_loss_vs_epochs', g_loss_epoch_ph)
    d_loss_epoch = tf.summary.scalar('d_loss_vs_epochs', d_loss_epoch_ph)
    epoch_summ = tf.summary.merge([g_loss_epoch, d_loss_epoch])

    summ_train_writer = tf.summary.FileWriter(log_path+'-train', tf.get_default_graph())
    summ_test_writer  = tf.summary.FileWriter(log_path+'-test',  tf.get_default_graph())

    with tf.Session() as sess:
        print('Training network ...')

        sess.run(init)

        print('Loading pre-trained network...', end=' ')
        #g_saver.restore(sess, '/'.join([model, 'pretrain', 'SRGAN_pretrain']))
        g_saver.restore(sess, 'model/20190714-214705_10_ua-va_wtk_us_2c/SRGAN00200/SRGAN') #'model/20190712-100906_10_ua-va_wtk_us_2c/'
        #gd_saver.restore(sess, 'model/20190714-214705_10_ua-va_wtk_us_2c/SRGAN00200/SRGAN')
        print('Done.')

        # Load perceptual loss network data if necessary
        if loss_type in ['AE', 'MSEandAE', 'VGG', 'MSEandVGG']:
            # Restore perceptual loss network, if necessary
            print('Loading perceptual loss network...', end=' ')
            var = tf.global_variables()
            if loss_type in ['AE', 'MSEandAE']:
                loss_var = [var_ for var_ in var if 'encoder' in var_.name]
            elif loss_type in ['VGG', 'MSEandVGG']:
                loss_var = [var_ for var_ in var if 'vgg19' in var_.name]
            saver = tf.train.Saver(var_list=loss_var)
            saver.restore(sess, loss_model)
            print('Done.')

        iters = 0
        for epoch in range(epoch_shift+1, epoch_shift+N_epochs+1):
            print('Epoch: '+str(epoch))
            start_time = time()

            # Loop through training data
            sess.run(init_iter_train)
            try:
                epoch_g_loss, epoch_d_loss, N_train = 0, 0, 0
                while True:
                    iters += 1
                    batch_idx, batch_LR, batch_HR, batch_lat, batch_lon = sess.run([idx, LR_out, HR_out, lat, lon])
                    #print("HR        ", np.min(batch_HR), np.mean(batch_HR), np.max(batch_HR))
                    batch_rough = roughnessGrabber(batch_lat, batch_lon, batch_HR.shape[1], batch_HR.shape[2], True)

                    feed_dict = {x_HR:batch_HR, x_LR:batch_LR, x_rough:batch_rough}
                    batch_SR = sess.run(model.x_SR, feed_dict=feed_dict) #try normalizing Surface.
                    #print(np.min(batch_SR), np.mean(batch_SR), np.max(batch_SR), batch_SR.shape)

                    #train the discriminator
                    sess.run(d_train_op, feed_dict={x_HR: batch_HR, x_LR: batch_LR})

                    #train the generator
                    sess.run(g_train_op, feed_dict=feed_dict)

                    #calculate current discriminator losses
                    gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict={x_HR: batch_HR, x_LR: batch_LR})

                    gen_count = 0
                    while (dl < 0.460) and gen_count < 30:
                        #discriminator did too well. train the generator extra
                        #print("generator training. discriminator loss : ", dl, "count : ", gen_count)
                        sess.run(g_train_op, feed_dict=feed_dict)
                        gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict={x_HR: batch_HR, x_LR: batch_LR})
                        gen_count += 1

                    dis_count = 0
                    while (dl >= 0.6) and dis_count <30:
                        #generator fooled the discriminator. train the discriminator extra
                        #print("discriminator training. discriminator loss : ", dl, "count : ", dis_count)
                        sess.run(d_train_op, feed_dict={x_HR: batch_HR, x_LR: batch_LR})
                        gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict={x_HR: batch_HR, x_LR: batch_LR})
                        dis_count += 1

                    pl, gal = sess.run([model.p_loss, model.g_ad_loss], feed_dict={x_HR: batch_HR, x_LR: batch_LR})
                    summ = sess.run(iter_summ, feed_dict={g_loss_iters_ph: gl, d_loss_iters_ph: dl})
                    summ_train_writer.add_summary(summ, iters)

                    epoch_g_loss += gl*N_batch
                    epoch_d_loss += dl*N_batch
                    N_train += N_batch
                    if (iters % print_every) == 0:
                        print('G loss=%.5f, Perceptual component=%.5f, Adversarial component=%.5f' %(gl, np.mean(pl), np.mean(gal)))
                        print('D loss=%.5f' %(dl))
                        print('TP=%.5f, TN=%.5f, FP=%.5f, FN=%.5f' %(p[0], p[1], p[2], p[3]))
                        print('')

            except tf.errors.OutOfRangeError:
                if (epoch % save_every) == 0:
                    model_epoch = '/'.join([model_name, 'SRGAN{0:05d}'.format(epoch)])
                    if not os.path.exists(model_epoch):
                        os.makedirs(model_epoch)
                    g_saver.save(sess, '/'.join([model_epoch, 'SRGAN']))
                g_loss_train = epoch_g_loss/N_train
                d_loss_train = epoch_d_loss/N_train

            # Loop through test data
            sess.run(init_iter_test)
            try:
                test_out = None
                epoch_g_loss, epoch_d_loss, N_test = 0, 0, 0
                while True:
                    batch_idx, batch_LR, batch_HR, batch_lat, batch_lon = sess.run([idx, LR_out, HR_out, lat, lon])

                    batch_rough = roughnessGrabber(batch_lat, batch_lon, batch_HR.shape[1], batch_HR.shape[2], True)
                    N_batch = batch_LR.shape[0]

                    feed_dict = {x_HR:batch_HR, x_LR:batch_LR, x_rough:batch_rough}

                    batch_SR, gl, dl = sess.run([model.x_SR, model.g_loss, model.d_loss], feed_dict=feed_dict)


                    epoch_g_loss += gl*N_batch
                    epoch_d_loss += dl*N_batch
                    N_test += N_batch
                    batch_LR = mu_sig[1]*batch_LR + mu_sig[0]
                    batch_SR = mu_sig[1]*batch_SR + mu_sig[0]
                    batch_HR = mu_sig[1]*batch_HR + mu_sig[0]
                    if (epoch % save_every) == 0:
                        nan_arr = np.zeros((batch_HR.shape[1], 10), dtype=np.float32)
                        nan_arr.fill(np.nan)
                        nan_arr2 = np.zeros((10, 3*batch_HR.shape[2] + 20), dtype=np.float32)
                        nan_arr2.fill(np.nan)

                        #for i, idx in enumerate(batch_LR.shape[0]): # NEED TO INDEX ALL THE DATA
                        for i, b_idx in enumerate(batch_idx):
                            idx_path = '/'.join([img_path, 'train', 'img{0:04d}'.format(b_idx)])
                            if not os.path.exists(idx_path):
                                os.makedirs(idx_path)

                            ua_img = np.concatenate((np.repeat(np.repeat(batch_LR[i, :, :, 0], np.prod(r), axis=0), np.prod(r), axis=1), \
                                                                          nan_arr, \
                                                                          batch_SR[i, :, :, 0], \
                                                                          nan_arr, \
                                                                          batch_HR[i, :, :, 0]), axis=1)
                            va_img = np.concatenate((np.repeat(np.repeat(batch_LR[i, :, :, 1], np.prod(r), axis=0), np.prod(r), axis=1), \
                                                                          nan_arr, \
                                                                          batch_SR[i, :, :, 1], \
                                                                          nan_arr, \
                                                                          batch_HR[i, :, :, 1]), axis=1)
                            cat_img = np.concatenate((ua_img, nan_arr2, va_img), axis=0)
                            plt.imsave('/'.join([idx_path, 'epoch{0:05d}'.format(epoch)+'.png']), cat_img, cmap='jet', format='png', origin='lower')
                            if epoch = epoch_shift+1:
                                plt.imsave('/'.join([idx_path, 'epoch{0:05d}'.format(epoch)+'surface.png']), batch_rough[i,:,:,0], cmap='Greys', format='png', origin='lower')
                            if test_out is None:
                                test_out = np.expand_dims(batch_SR[i], 0)
                            else:
                                test_out = np.concatenate((test_out, np.expand_dims(batch_SR[i], 0)), axis=0)

            except tf.errors.OutOfRangeError:

                g_loss_test = epoch_g_loss/N_test
                d_loss_test = epoch_d_loss/N_test

                if not os.path.exists(test_data_path):
                    os.makedirs(test_data_path)
                if (epoch % save_every) == 0:
                    np.save(test_data_path +'/WTK_test_SR_epoch{0:05d}'.format(epoch)+'.npy', test_out)


                # Write performance to TensorBoard
                summ = sess.run(epoch_summ, feed_dict={g_loss_epoch_ph: g_loss_train, d_loss_epoch_ph: d_loss_train})
                summ_train_writer.add_summary(summ, epoch)

                summ = sess.run(epoch_summ, feed_dict={g_loss_epoch_ph: g_loss_test, d_loss_epoch_ph: d_loss_test})
                summ_test_writer.add_summary(summ, epoch)

                print('Epoch took %.2f seconds\n' %(time() - start_time))

        if not os.path.exists(model_name):
            os.makedirs(model_name)
        g_saver.save(sess, '/'.join([model_name, 'SRGAN', 'SRGAN']))
        gd_saver.save(sess, '/'.join([model_name, 'SRGAN-all', 'SRGAN']))

    print('Done.')


def test(mu_sig, isCCSM):
    """Run test data through generator and save output."""

    print('Initializing network ...', end=' ')
    tf.reset_default_graph()
    # Set low-res data place holders
    x_LR = tf.placeholder(tf.float32, [None, None, None, 2])

    # Set super resolution scaling. Needs to match network architecture
    r = [2, 5]
    SR_scale = r[0]*r[1]
    # Initialize network and set optimizer
    model = SRGAN(x_LR=x_LR, r=r, status='testing')
    init = tf.global_variables_initializer()
    g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
    print('Done.')

    print('Building data pipeline ...', end=' ')

    ds_test = tf.data.TFRecordDataset(ccsm_test_path)
    ds_test = ds_test.map(lambda xx: _parse_CCSM_(xx, mu_sig)).batch(batch_size)

    iterator = tf.data.Iterator.from_structure(ds_test.output_types,
                                               ds_test.output_shapes)
    idx, LR_out, lat, lon = iterator.get_next()
    print(idx)
    init_iter_test  = iterator.make_initializer(ds_test)
    print('Done.')

    # Set expected size of SR data.
    with tf.Session() as sess:
        print('Loading saved network ...', end=' ')
        sess.run(init)

        # Load trained model
        #g_saver.restore(sess, '/'.join([model_name, 'pretrain', 'SRGAN_pretrain']))
        #g_saver.restore(sess, '/'.join([model_name, 'SRGAN', 'SRGAN']))
        #g_saver.restore(sess, 'model/20190712-100906_10_ua-va_wtk_us_2c/pretrain02000/SRGAN_pretrain')
        g_saver.restore(sess, 'model/20191016-193307_10_ua-va_wtk_us_2c/pretrain00929/SRGAN')
        print('Done.')

        print('Running test data ...')
        # Loop through test data
        sess.run(init_iter_test)
        try:
            data_out = None
            while True:
                print(idx, LR_out.shape)
                batch_idx, batch_LR, batch_lat, batch_lon = sess.run([idx, LR_out])
                batch_rough = roughnessGrabber(batch_lat, batch_lon, batch_HR.shape[1]*SR_scale, batch_HR.shape[2]*SR_scale, True)
                batch_SR = sess.run(model.x_SR, feed_dict={x_LR: batch_LR, x_rough: batch_rough})

                batch_SR = mu_sig[1]*batch_SR + mu_sig[0]
                if data_out is None:
                    data_out = batch_SR
                else:
                    data_out = np.concatenate((data_out, batch_SR), axis=0)

        except tf.errors.OutOfRangeError:
            pass

        if not os.path.exists(test_data_path):
            os.makedirs(test_data_path)
        np.save(test_data_path+'/CCSM_test_SR.npy', data_out)

    print('Done.')


# Parser function for data pipeline. May need alternative parser for tfrecords without high-res counterpart
def _parse_WTK_(serialized_example, mu_sig=None):
    feature = {'index': tf.FixedLenFeature([], tf.int64),
             'data_LR': tf.FixedLenFeature([], tf.string),
                'h_LR': tf.FixedLenFeature([], tf.int64),
                'w_LR': tf.FixedLenFeature([], tf.int64),
             'data_HR': tf.FixedLenFeature([], tf.string),
                'h_HR': tf.FixedLenFeature([], tf.int64),
                'w_HR': tf.FixedLenFeature([], tf.int64),
                   'c': tf.FixedLenFeature([], tf.int64),
                 'lat': tf.FixedLenFeature([], tf.string),
                 'lon': tf.FixedLenFeature([], tf.string)}
    example = tf.parse_single_example(serialized_example, feature)

    idx = example['index']

    h_LR, w_LR = example['h_LR'], example['w_LR']
    h_HR, w_HR = example['h_HR'], example['w_HR']

    c = example['c']

    lat, lon = tf.decode_raw(example['lat'], tf.float64), tf.decode_raw(example['lon'], tf.float64)

    data_LR = tf.decode_raw(example['data_LR'], tf.float64)
    data_HR = tf.decode_raw(example['data_HR'], tf.float64)

    data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))
    data_HR = tf.reshape(data_HR, (h_HR, w_HR, c))

    if mu_sig is not None:
        data_LR = (data_LR - mu_sig[0])/mu_sig[1]
        data_HR = (data_HR - mu_sig[0])/mu_sig[1]

    return idx, data_LR, data_HR, lat, lon

def _parse_CCSM_(serialized_example, mu_sig=None):
    feature = {'index': tf.FixedLenFeature([], tf.int64),
             'data_LR': tf.FixedLenFeature([], tf.string),
                'h_LR': tf.FixedLenFeature([], tf.int64),
                'w_LR': tf.FixedLenFeature([], tf.int64),
                   'c': tf.FixedLenFeature([], tf.int64),
                 'lat': tf.FixedLenFeature([], tf.string),
                 'lon': tf.FixedLenFeature([], tf.string)}
    example = tf.parse_single_example(serialized_example, feature)

    idx = example['index']

    h_LR, w_LR = example['h_LR'], example['w_LR']

    c = example['c']

    lat, lon = tf.decode_raw(example['lat'], tf.float32), tf.decode_raw(example['lon'], tf.float32)

    data_LR = tf.decode_raw(example['data_LR'], tf.float32)
    data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))

    if mu_sig is not None:
        data_LR = (data_LR - mu_sig[0])/mu_sig[1]
    return idx, data_LR, lat, lon

if __name__ == '__main__':
    # Compute mu, sigma for all channels of data
    # NOTE: This includes building a temporary data pipeline and looping through everything.
    #       There's probably a smarter way to do this...
    print('Loading data ...', end=' ')
    dataset = tf.data.TFRecordDataset(train_path)
    dataset = dataset.map(_parse_WTK_).batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    _, _, HR_out,_,_ = iterator.get_next()
    print(HR_out.shape)
    with tf.Session() as sess:
        N, mu, sigma = 0, 0, 0
        try:
            while True:
                data_HR = sess.run(HR_out)
                #print(data_HR.shape, np.min(data_HR),np.mean(data_HR),np.max(data_HR))
                N_batch, h, w, c = data_HR.shape
                N_new = N + N_batch
                mu_batch = np.mean(data_HR, axis=(0, 1, 2))
                sigma_batch = np.var(data_HR, axis=(0, 1, 2))

                sigma = (N/N_new)*sigma + (N_batch/N_new)*sigma_batch + (N*N_batch/N_new**2)*(mu - mu_batch)**2
                mu = (N/N_new)*mu + (N_batch/N_new)*mu_batch

                N = N_new

        except Exception as e:#tf.errors.OutOfRangeError:
            #print(e)
            #print("error")
            pass
    mu_sig = [mu, np.sqrt(sigma)]
    print('Done.')

    #pre_train(mu_sig)
    #train(mu_sig)
    test(mu_sig, False)