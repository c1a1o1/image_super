import os
import tensorflow as tf
import numpy as np
import numpy.random
import time
from op import *
from utils import  *
from model_until import  *

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

batch_size = 16
dim = 64
k = 4 # downscale size
gene_l1_factor = 0.6
beta1 = 0.5
learning_rate_start = 0.0002
learning_rate_half_life = 5000
dataset = '/home/data/houruibing/CelebA/img_align_celeba'
test_vector = 16 # num of test images
checkpoint_period = 10000
summary_period = 200
train_time = 60 * 3 # time in minutes to train the model

#prepare data
filenames = tf.gfile.ListDirectory(dataset)
filenames = sorted(filenames)
random.shuffle(filenames)
filenames = [os.path.join(dataset, f) for f in filenames]
train_filenames = filenames[: -test_vector]
test_filenames = filenames[-test_vector: ]

#checkpoint_dir = 'checkpoint/CelebA'
#if not os.path.exists(checkpoint_dir):
#    os.makedirs(checkpoint_dir)

#train_dir = 'generate_images/CelebA'
#if not os.path.exists(train_dir):
#    os.makedirs(train_dir)
    
checkpoint_dir = 'checkpoint/0.6_l1_loss/CelebA'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

train_dir = 'generate_images/0.6_l1_loss/CelebA'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)


def discriminator(image, reuse = False):
    """

    :param image: a batch of HR image or generate HR image [batch_size, 64, 64, 3]
    :return: A tensor that represents the probability of the real image[batch_size, 1]
    """
    with tf.variable_scope("Disc") as scope:

        if reuse:
            scope.reuse_variables()
        image = 2 * image - 1
        h1 = relu(batch_norm(conv2d(image, dim, name = 'd_h1_conv'), name = 'd_bn1'))
        h2 = relu(batch_norm(conv2d(h1, 2 * dim, name = 'd_h2_conv'), name = 'd_bn2'))
        h3 = relu(batch_norm(conv2d(h2, 4 * dim, name = 'd_h3_conv'), name = 'd_bn3'))
        h4 = relu(batch_norm(conv2d(h3, 8 * dim, name = 'd_h4_conv'), name = 'd_bn4'))
        h5 = relu(batch_norm(conv2d(h4, 8 * dim, k_h = 3, k_w = 3, d_h = 1, d_w = 1, name= 'd_h5_conv'), name = 'd_bn5'))
        h6 = relu(batch_norm(conv2d(h5, 8 * dim, k_h = 1, k_w = 1, d_h = 1, d_w = 1,name = 'd_h6_conv'), name = 'd_bn6'))
        h7 = relu(batch_norm(conv2d(h6, 1, 1, 1, 1, 1, name = 'd_h7_conv'), name = 'd_bn7'))
        y = mean(h7)
        return y

def generator(LR, reuse = False):
    """

    :param LR: the input LR image [batch_size, 16, 16, 3]
    :return: the generate HR image [batch_size, 64, 64, 3]
    """
    with tf.variable_scope("gen") as scope:
        if reuse:
            scope.reuse_variables()
        h1 = conv2d(LR, dim * 4, k_h = 1, k_w = 1, d_h = 1, d_w = 1, stddev = 1.0, name = "g_h1_conv")
        h2 = residual_block(h1, dim * 4, name = "g_h2_res")
        h3 = residual_block(h2, dim * 4, name = "g_h3_res")
        h4 = relu(batch_norm(upscale(h3), name = "g_h4_upscale"))
        h5 = deconv2d(h4, [batch_size, 32, 32, 256], 3, 3, 1, 1, name = "g_h5_deconv")
        h6 = conv2d(h5, dim * 2, k_h = 1, k_w = 1, d_h = 1, d_w = 1, name = "g_h6_conv")
        h7 = residual_block(h6, dim * 2, name = "g_h7_res")
        h8 = residual_block(h7, dim * 2, name = "g_h8_res")
        h9 = relu(batch_norm(upscale(h8), name = "g_h9_upscale"))
        h10 = deconv2d(h9, [batch_size, 64, 64, 128], 3, 3, 1, 1, name = "g_h10_deconv")
        h11 = relu(conv2d(h10, 96, d_h = 1, d_w = 1, name = "g_h11_conv"))
        h12 = relu(conv2d(h11, 96, k_h = 1, k_w = 1, d_h = 1, d_w = 1, name = "g_h12_conv"))
        output = sigmoid(conv2d(h12, 3, k_h = 1, k_w = 1, d_h = 1, d_w = 1, stddev = 1.0, name = "g_out_conv"))
        return output

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

# setup async input queues
train_features, train_labels = setup_inputs(sess, 64, train_filenames, batch_size)
test_features, test_labels = setup_inputs(sess, 64, test_filenames, batch_size)

# add some noise during training
noise_level = 0.03
noise_train_features = gaussian_noise_layer(train_features, std=noise_level)

#create model
gene_output = generator(noise_train_features)

gene_minput = tf.placeholder(tf.float32, [None, 16, 16, 3])
gene_moutput = generator(gene_minput, reuse = True)
#disc_real_input = tf.placeholder(tf.float32, [None, 64, 64, 3])
disc_real_output = discriminator(tf.identity(train_labels))
disc_fake_output = discriminator(gene_output, reuse = True)

#generator loss
#feature = tf.placeholder(tf.float32, [None, 16, 16, 3])
cross_entropy_gen = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_output, labels=tf.ones_like(disc_fake_output))
gene_ce_loss = tf.reduce_mean(cross_entropy_gen)
# does the result look like the feature(L1 loss)
downscaled = downscale(gene_output, k)
gene_l1_loss = tf.reduce_mean(tf.abs(downscaled - train_features))
gen_loss = (1.0 - gene_l1_factor) * gene_ce_loss + gene_l1_factor * gene_l1_loss

#discriminator loss
cross_entropy_dis_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_output, labels=tf.ones_like(disc_real_output))
disc_real_loss = tf.reduce_mean(cross_entropy_dis_real)
cross_entropy_dis_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_output, labels=tf.zeros_like(disc_fake_output))
disc_fake_loss = tf.reduce_mean(cross_entropy_dis_fake)
disc_loss = disc_real_loss + disc_fake_loss

#optimizers
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]
learning_rate = tf.placeholder(dtype = tf.float32, name='learning_rate')
d_optim = tf.train.AdamOptimizer(learning_rate, beta1).minimize(disc_loss, var_list = d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate, beta1).minimize(gen_loss, var_list = g_vars)

test_feature, test_label = sess.run([test_features, test_labels])

#train
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
lrval = learning_rate_start
start_time = time.time() # seconds
done = False
batch = 0

while not done:
    batch += 1
    ops = [d_optim, g_optim, disc_real_loss, disc_fake_loss, disc_loss, gene_ce_loss, gene_l1_loss, gen_loss]
    feed_dict = {learning_rate: lrval}
    _, _, Disc_real_loss, Disc_fake_loss, Disc_loss, Gene_ce_loss, Gene_l1_loss, Gen_loss = sess.run(ops, feed_dict=feed_dict)
    if batch % 10 == 0:
        elapsed = int(time.time() - start_time) / 60 # have train minutes
        print 'Progress[%3d%%], ETA[%4dm], Batch [%4d], Gene_ce_loss[%3.3f], Gene_l1_loss[%3.3f], Gen_loss[%3.3f], D_real_loss[%3.3f], D_Fake_Loss[%3.3f], D_loss[%3.3f]' %  (int(100 * elapsed / train_time), train_time - elapsed, batch, Gene_ce_loss, Gene_l1_loss, Gen_loss, Disc_real_loss, Disc_fake_loss, Disc_loss)

        #finished?
        current_progress = elapsed / train_time
        if current_progress >= 1.0:
            done = True

        if batch % learning_rate_half_life == 0:
            lrval *= 0.5

    if batch % checkpoint_period == 0:
        save(checkpoint_dir, saver, sess, batch)

    if batch % summary_period == 0:
        feed_dict = {gene_minput: test_feature}
        gene_test_output = sess.run(gene_moutput, feed_dict=feed_dict)
        summarize_progress(sess, train_dir, test_feature, test_label, gene_test_output, batch, 'out')











