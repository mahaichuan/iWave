from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

################ The priori parameters of iWave

growth_rate = 12

decomposition_step = 4

#################  The basic DL operations

def weight_variable(name,shape,std):

  initial = tf.truncated_normal(shape, stddev=std)

  return tf.get_variable(name, initializer=initial)


def conv2d(x, W, stride=1):

  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def conv2d_pad(x, W, stride=1):

    paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])

    x = tf.pad(x, paddings=paddings,mode="REFLECT")

    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

############### The basic lifting operations, P and U steps

def preprocessing(x):

    x_shape = x.get_shape().as_list()

    x_c = x_shape[3]

    w = weight_variable('conv1',[3,3,x_c,int(2*growth_rate)],0.01)

    c = conv2d_pad(x,w,1)

    return c

def denset_layer(x):

    x_shape = x.get_shape().as_list()

    x_c = x_shape[3]

    c = tf.tanh(x)

    w = weight_variable('conv1',[1,1,x_c,int(4*growth_rate)],0.01)

    c = conv2d(c,w,1)

    c = tf.tanh(c)

    w = weight_variable('conv2',[3, 3, int(4*growth_rate), growth_rate], 0.01)

    c = conv2d_pad(c, w, 1)

    return c

def denset_block(x,L):

    for i in range(L):

        with tf.variable_scope('block_'+str(i)):

            c = denset_layer(x)

            x = tf.concat([x, c], 3)

    return x

def postprocessing(x):

    x_shape = x.get_shape().as_list()

    x_c = x_shape[3]

    c = tf.tanh(x)

    w = weight_variable('conv1',[1, 1, x_c, 1], 0.01)

    c = conv2d(c,w,1)

    return c

def PQ(x):

    with tf.variable_scope('preprocess'):
        x = preprocessing(x)

    with tf.variable_scope('dense_block'):

        x = denset_block(x, 4)

    with tf.variable_scope('postprocess'):

        x = postprocessing(x)

    return x

def UQ(L, H):

    L = (L + H) / 2

    return L

def dUQ(L, H):

    L = 2 * L - H

    return L

######################## The lifting scheme, update first, acrodding to Vim's non-linear theory

def decomposition(x):

    # step 1: for h
    L = x[:, 0::2, :, :]

    H = x[:, 1::2, :, :]

    L = UQ(L,H)

    with tf.variable_scope('PQ', reuse=tf.AUTO_REUSE):

        H = H - PQ(L)

    # step 2: for w, L

    L = tf.transpose(L, [0, 2, 1, 3])

    LL = L[:, 0::2, :, :]

    HL = L[:, 1::2, :, :]

    LL = UQ(LL, HL)

    with tf.variable_scope('PQ', reuse=tf.AUTO_REUSE):

        HL = HL - PQ(LL)

    LL = tf.transpose(LL, [0, 2, 1, 3])

    HL = tf.transpose(HL, [0, 2, 1, 3])

    # step 2: for w, H

    H = tf.transpose(H, [0, 2, 1, 3])

    LH = H[:, 0::2, :, :]

    HH = H[:, 1::2, :, :]

    LH = UQ(LH, HH)

    with tf.variable_scope('PQ', reuse=tf.AUTO_REUSE):

        HH = HH - PQ(LH)

    LH = tf.transpose(LH, [0, 2, 1, 3])

    HH = tf.transpose(HH, [0, 2, 1, 3])

    return LL, HL, LH, HH

def reconstruct_fun(up,bot):

    temp_L = tf.transpose(up, [0, 2, 1, 3])

    temp_H = tf.transpose(bot, [0, 2, 1, 3])

    x_shape = temp_L.get_shape().as_list()

    x_n = x_shape[0]
    x_h = x_shape[1]
    x_w = x_shape[2]
    x_c = x_shape[3]

    temp_L = tf.reshape(temp_L, [x_n, x_h * x_w, 1, x_c])

    temp_H = tf.reshape(temp_H, [x_n, x_h * x_w, 1, x_c])

    temp = tf.concat([temp_L,temp_H],2)

    temp = tf.reshape(temp,[x_n, x_h, 2*x_w, x_c])

    recon = tf.transpose(temp, [0, 2, 1, 3])

    return recon

def reconstruct(LL,HL,LH,HH):

    LL = tf.transpose(LL, [0, 2, 1, 3])

    HL = tf.transpose(HL, [0, 2, 1, 3])

    with tf.variable_scope('PQ', reuse=tf.AUTO_REUSE):

        HL = HL + PQ(LL)

    LL = dUQ(LL, HL)

    L = reconstruct_fun(LL,HL)

    L = tf.transpose(L, [0, 2, 1, 3])

    LH = tf.transpose(LH, [0, 2, 1, 3])

    HH = tf.transpose(HH, [0, 2, 1, 3])

    with tf.variable_scope('PQ', reuse=tf.AUTO_REUSE):

        HH = HH + PQ(LH)

    LH = dUQ(LH, HH)

    H = reconstruct_fun(LH, HH)

    H = tf.transpose(H, [0, 2, 1, 3])

    with tf.variable_scope('PQ', reuse=tf.AUTO_REUSE):

        H = H + PQ(L)

    L = dUQ(L, H)

    recon = reconstruct_fun(L, H)

    return recon

def reconstruct_all(LL, HL, LH, HH, layer):

    LL = reconstruct(LL,HL,LH,HH)

    for i in range(layer):

        HL = tf.constant(0., dtype=tf.float32, shape=LL.get_shape().as_list())
        LH = tf.constant(0., dtype=tf.float32, shape=LL.get_shape().as_list())
        HH = tf.constant(0., dtype=tf.float32, shape=LL.get_shape().as_list())

        LL = reconstruct(LL, HL, LH, HH)

    recon = LL

    return recon

def reconstruct_coeff(coeff_list):

    LL = coeff_list[len(coeff_list) - 1]

    for j in range(1, len(coeff_list), 3):

        i = len(coeff_list) - 1 - j

        HL = coeff_list[i]
        LH = coeff_list[i - 1]
        HH = coeff_list[i - 2]

        up = tf.concat([LL, HL], 2)
        bt = tf.concat([LH, HH], 2)

        LL = tf.concat([up, bt], 1)

    coeff = LL

    x_shape = coeff.get_shape().as_list()

    x_n = x_shape[0]
    x_h = x_shape[1]
    x_w = x_shape[2]
    x_c = x_shape[3]

    coeff = tf.reshape(coeff,[x_h,x_w])

    return coeff

gain_norm = [255.031942246499,	259.052257134387,	262.902693670905,	275.771414487117,	281.032529341253,	286.721550993854,	309.372176558730,	305.822216964638,	316.651157856070,	344.618613618763,	333.281296675879,	343.539889010862,	342.742664864807]

def graph(x):

    x1 = 0.299 * x[:,:,:,0:1] + 0.587 * x[:,:,:,1:2] + 0.114 * x[:,:,:,2:3]
    x2 = -0.16875 * x[:, :, :, 0:1] -0.33126 * x[:, :, :, 1:2] + 0.5 * x[:, :, :, 2:3]
    x3 = 0.5 * x[:, :, :, 0:1]  -0.41869 * x[:, :, :, 1:2] -0.08131 * x[:, :, :, 2:3]

    LL = x1

    HL_collection = []
    LH_collection = []
    HH_collection = []

    coeff_list = []

    gain_id = 0

    for i in range(decomposition_step):

        with tf.variable_scope('decomposition', reuse=tf.AUTO_REUSE):

            LL, HL, LH, HH = decomposition(LL)

            coeff_list.append(HH * gain_norm[gain_id])
            gain_id = gain_id + 1

            coeff_list.append(LH * gain_norm[gain_id])
            gain_id = gain_id + 1

            coeff_list.append(HL * gain_norm[gain_id])
            gain_id = gain_id + 1

        HL_collection.append(HL)
        LH_collection.append(LH)
        HH_collection.append(HH)

    coeff_list.append(LL * gain_norm[gain_id])

    coeff1 = reconstruct_coeff(coeff_list)

    LL = x2

    HL_collection = []
    LH_collection = []
    HH_collection = []

    coeff_list = []

    gain_id = 0

    for i in range(decomposition_step):

        with tf.variable_scope('decomposition', reuse=tf.AUTO_REUSE):

            LL, HL, LH, HH = decomposition(LL)

            coeff_list.append(HH * gain_norm[gain_id])
            gain_id = gain_id + 1

            coeff_list.append(LH * gain_norm[gain_id])
            gain_id = gain_id + 1

            coeff_list.append(HL * gain_norm[gain_id])
            gain_id = gain_id + 1

        HL_collection.append(HL)
        LH_collection.append(LH)
        HH_collection.append(HH)

    coeff_list.append(LL * gain_norm[gain_id])

    coeff2 = reconstruct_coeff(coeff_list)

    LL = x3

    HL_collection = []
    LH_collection = []
    HH_collection = []

    coeff_list = []

    gain_id = 0

    for i in range(decomposition_step):

        with tf.variable_scope('decomposition', reuse=tf.AUTO_REUSE):

            LL, HL, LH, HH = decomposition(LL)

            coeff_list.append(HH * gain_norm[gain_id])
            gain_id = gain_id + 1

            coeff_list.append(LH * gain_norm[gain_id])
            gain_id = gain_id + 1

            coeff_list.append(HL * gain_norm[gain_id])
            gain_id = gain_id + 1

        HL_collection.append(HL)
        LH_collection.append(LH)
        HH_collection.append(HH)

    coeff_list.append(LL * gain_norm[gain_id])

    coeff3 = reconstruct_coeff(coeff_list)

    return coeff1, coeff2, coeff3

parser = argparse.ArgumentParser(description='iWave forward transfrom...')

parser.add_argument('--height', type=int, default=512, help="height")

parser.add_argument('--width', type=int, default=768, help="width")

parser.add_argument('--img', default=None, help='your_image_path, only RGB image is valid!')

opt = parser.parse_args()

w_in = opt.width

h_in = opt.height

stride = w_in * h_in

x = tf.placeholder(tf.float32, [1, h_in, w_in, 3])

coeff1, coeff2, coeff3 = graph(x)

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess,'./my-model-439999')

    coeff_yuv = np.zeros((3 * stride, 1))

    img = Image.open(opt.img)

    img = np.asarray(img, dtype=np.float32)
    img = img / 255 - 0.5
    img = np.reshape(img, (1, h_in, w_in, 3))

    coeff1_eval, coeff2_eval, coeff3_eval = sess.run([coeff1, coeff2, coeff3], feed_dict={x: img})

    coeff_yuv[0:stride] = np.round(np.reshape(coeff1_eval, (stride, 1)) * 4000.)

    coeff_yuv[stride:2 * stride] = np.round(np.reshape(coeff2_eval, (stride, 1)) * 4000.)

    coeff_yuv[2 * stride:3 * stride] = np.round(np.reshape(coeff3_eval, (stride, 1)) * 4000.)

    np.savetxt('./coeffs/tmp.txt', coeff_yuv, fmt='%d')

    sess.close()
