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

def depart_coeff(coeff):

    x_shape = coeff.get_shape().as_list()

    x_h = x_shape[0]
    x_w = x_shape[1]

    HL_collection = []
    LH_collection = []
    HH_collection = []

    for i in range(decomposition_step):

        HL = coeff[0:int(x_h/2), int(x_w/2):x_w]
        HL = tf.reshape(HL, [1,int(x_h/2), int(x_w/2),1])

        LH = coeff[int(x_h / 2):x_h, 0:int(x_w / 2)]
        LH = tf.reshape(LH, [1, int(x_h / 2), int(x_w / 2), 1])

        HH = coeff[int(x_h / 2):x_h, int(x_w / 2):x_w]
        HH = tf.reshape(HH, [1, int(x_h / 2), int(x_w / 2), 1])

        HL_collection.append(HL)
        LH_collection.append(LH)
        HH_collection.append(HH)

        coeff = coeff[0:int(x_h / 2), 0:int(x_w / 2)]

        x_shape = coeff.get_shape().as_list()

        x_h = x_shape[0]
        x_w = x_shape[1]

    LL = coeff

    LL = tf.reshape(LL, [1, x_h, x_w, 1])

    return LL, HL_collection, LH_collection, HH_collection


gain_norm = [255.031942246499,	259.052257134387,	262.902693670905,	275.771414487117,	281.032529341253,	286.721550993854,	309.372176558730,	305.822216964638,	316.651157856070,	344.618613618763,	333.281296675879,	343.539889010862,	342.742664864807]

def graph(coeff1, coeff2, coeff3):

    # ////////////////  load coeff1

    coeff = coeff1

    LL, HL_collection, LH_collection, HH_collection = depart_coeff(coeff)

    # ////////////  i dwt

    gain_id = 0

    for i in range(decomposition_step):

        HH_collection[i] = HH_collection[i] / gain_norm[gain_id]
        gain_id = gain_id + 1

        LH_collection[i] = LH_collection[i] / gain_norm[gain_id]
        gain_id = gain_id + 1

        HL_collection[i] = HL_collection[i] / gain_norm[gain_id]
        gain_id = gain_id + 1

    LL = LL / gain_norm[gain_id]

    for j in range(decomposition_step):

        i = decomposition_step - 1 - j

        with tf.variable_scope('decomposition', reuse=tf.AUTO_REUSE):

            LL = reconstruct(LL, HL_collection[i], LH_collection[i], HH_collection[i])

    y = LL

    # ////////////////  load coeff2

    coeff = coeff2

    LL, HL_collection, LH_collection, HH_collection = depart_coeff(coeff)

    # ////////////  i dwt

    gain_id = 0

    for i in range(decomposition_step):

        HH_collection[i] = HH_collection[i] / gain_norm[gain_id]
        gain_id = gain_id + 1

        LH_collection[i] = LH_collection[i] / gain_norm[gain_id]
        gain_id = gain_id + 1

        HL_collection[i] = HL_collection[i] / gain_norm[gain_id]
        gain_id = gain_id + 1

    LL = LL / gain_norm[gain_id]

    for j in range(decomposition_step):

        i = decomposition_step - 1 - j

        with tf.variable_scope('decomposition', reuse=tf.AUTO_REUSE):

            LL = reconstruct(LL, HL_collection[i], LH_collection[i], HH_collection[i])

    u = LL

    # ////////////////  load coeff3

    coeff = coeff3

    LL, HL_collection, LH_collection, HH_collection = depart_coeff(coeff)

    # ////////////  i dwt

    gain_id = 0

    for i in range(decomposition_step):

        HH_collection[i] = HH_collection[i] / gain_norm[gain_id]
        gain_id = gain_id + 1

        LH_collection[i] = LH_collection[i] / gain_norm[gain_id]
        gain_id = gain_id + 1

        HL_collection[i] = HL_collection[i] / gain_norm[gain_id]
        gain_id = gain_id + 1

    LL = LL / gain_norm[gain_id]

    for j in range(decomposition_step):

        i = decomposition_step - 1 - j

        with tf.variable_scope('decomposition', reuse=tf.AUTO_REUSE):

            LL = reconstruct(LL, HL_collection[i], LH_collection[i], HH_collection[i])

    v = LL

    # -------------------------------------------------

    r = y + 1.402*v
    g = y - 0.34413*u - 0.71414*v
    b = y + 1.772*u

    recon = tf.concat([r,g,b], axis=3)

    recon = (recon + 0.5) * 255

    recon = tf.clip_by_value(recon, 0., 255.)

    recon = tf.round(recon)

    return recon



parser = argparse.ArgumentParser(description='iWave forward transfrom...')

parser.add_argument('--height', type=int, default=512, help="height")

parser.add_argument('--width', type=int, default=768, help="width")

parser.add_argument('--recon_img', default=None, help='your_recon_path')

opt = parser.parse_args()

w_in = opt.width

h_in = opt.height

stride = w_in*h_in

coeff1 = tf.placeholder(tf.float32, [h_in, w_in])
coeff2 = tf.placeholder(tf.float32, [h_in, w_in])
coeff3 = tf.placeholder(tf.float32, [h_in, w_in])

recon = graph(coeff1, coeff2, coeff3)


saver = tf.train.Saver()

mse_all = []


with tf.Session() as sess:

    saver.restore(sess, './my-model-439999')

    # load coeff
    coeff_yuv = np.loadtxt('./recon_coeffs/tmp.txt')
    coeff_yuv = np.asarray(coeff_yuv, dtype=np.float32) / 4000.

    mat_y = coeff_yuv[0:stride]
    mat_u = coeff_yuv[stride:2 * stride]
    mat_v = coeff_yuv[2 * stride:3 * stride]

    coeff_input_y = np.reshape(mat_y, (h_in, w_in))
    coeff_input_u = np.reshape(mat_u, (h_in, w_in))
    coeff_input_v = np.reshape(mat_v, (h_in, w_in))

    recon_eval = sess.run(recon, feed_dict={coeff1: coeff_input_y, coeff2: coeff_input_u, coeff3: coeff_input_v})

    recon_eval = recon_eval[0, :, :, :]

    img = Image.fromarray(np.uint8(recon_eval))

    img.save(opt.recon_img)

    sess.close()