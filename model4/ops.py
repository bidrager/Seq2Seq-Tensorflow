import tensorflow as tf
import numpy as np
import model4.config as config

import tensorflow.compat.v1 as tf #使用1版本
tf.disable_v2_behavior()

def norm_layer(name,x,train,eps=1e-5,decay=0.9):
    with tf.name_scope(name):
        params_shape = x.get_shape().as_list()
        param_shape = params_shape[-1:]
        scale = tf.get_variable( name+'_scales',param_shape, initializer=tf.constant_initializer(1.))
        offset = tf.get_variable(name+'_offsets', param_shape,initializer=tf.constant_initializer(0.0))
        moving_mean=tf.get_variable(name+'_mean',param_shape,initializer=tf.zeros_initializer,trainable=False)
        moving_variance=tf.get_variable(name+'_variance',param_shape,initializer=tf.zeros_initializer,trainable=False)

        mean, var = tf.nn.moments(x, axes=[0,1,2], name='moments')
        train_mean_op = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
        train_var_op = tf.assign(moving_variance, moving_variance * decay + var * (1 - decay))
        if train:
            with tf.control_dependencies([train_mean_op,train_var_op]):
                return tf.nn.batch_normalization(x,mean,var,offset,scale,eps)
        else:
            return tf.nn.batch_normalization(x,moving_mean,moving_variance,offset,scale,eps)

def create_weight(shape,name):
    w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32),
                           name=name,shape=shape)
    # tf.summary.scalar(name, tf.reduce_sum(w))
    # tf.add_to_collection('regularizer_losses', tf.contrib.layers.l2_regularizer(config.lambd)(w))
    return w
def create_bias(shape,name):
    b =tf.get_variable(initializer=tf.constant_initializer(value=0,dtype=tf.float32),
                           name='bias_'+name,shape=shape)
    # tf.summary.scalar(name, tf.reduce_sum(b))
    return b

def conv2d(input,name,filter_size,filer_num,train):
    with tf.name_scope(name):
        filter_deep=input.get_shape().as_list()[-1]
        conv1_w=create_weight(shape=[filter_size,filter_size,filter_deep,filer_num],name=name+'_w')
        conv1_b=create_bias([filer_num],name=name+'_b')
        conv1=tf.nn.conv2d(input,conv1_w,strides=[1,1,1,1],padding='SAME')
        conv=tf.nn.bias_add(conv1,conv1_b)
        conv=norm_layer(name+'_batch_norm',conv,train)
        return conv

def resNet(image,train):
    with tf.name_scope('resnet'):
        conv1 = tf.nn.relu(conv2d( image, 'conv1',5, config.conv_base_filter,train))
        conv2 = tf.nn.relu(conv2d(conv1, 'conv2', 3, config.conv_base_filter, train))
        X1=tf.concat([conv1,conv2],axis=3)
        X1= tf.nn.max_pool(name='pool1', value=X1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        conv3 = tf.nn.relu(conv2d(X1, 'conv3', 3,config.conv_base_filter*2,train))
        conv4 = tf.nn.relu(conv2d(conv3, 'conv4',3, config.conv_base_filter * 2, train))
        X2 = tf.concat([conv3, conv4], axis=3)
        X2 = tf.nn.max_pool(name='pool2',value= X2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

        conv5 = tf.nn.relu(conv2d(X2, 'conv5', 3, config.conv_base_filter *2, train))
        conv6= tf.nn.relu(conv2d(conv5, 'conv6',3, config.conv_base_filter * 2, train))
        X3 = tf.concat([conv5, conv6], axis=3)
        X3 = tf.nn.max_pool(name='pool3', value=X3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv7 = tf.nn.relu(conv2d(X3, 'conv7', 3, config.conv_base_filter *4, train))
        conv8 = tf.nn.relu(conv2d(conv7, 'conv8', 3, config.conv_base_filter * 4, train))
        X4 = tf.concat([conv7, conv8], axis=3)
        return X4


def im2latex_cnn(X,train):
    with tf.name_scope('im2latex_cnn'):
        X = tf.nn.relu(conv2d( X, 'conv1',3, config.conv_base_filter,train))
        X = tf.nn.max_pool(name='pool1', value=X, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        X = tf.nn.relu(conv2d(X, 'conv2', config.conv_base_filter,config.conv_base_filter*2,train))
        X = tf.nn.max_pool(name='pool2',value= X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

        X = tf.nn.relu(conv2d(X,'conv3', config.conv_base_filter*2,config.conv_base_filter*4,train))

        X = tf.nn.relu(conv2d(X, 'conv4', config.conv_base_filter*4,config.conv_base_filter*4,train))
        X = tf.nn.max_pool(name='pool4',value= X, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],padding='SAME')

        X = tf.nn.relu(conv2d(X, 'conv5', config.conv_base_filter * 4, config.conv_base_filter * 8,train))
        X = tf.nn.max_pool(name='pool5', value=X, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],padding='SAME')

        X = tf.nn.relu(conv2d(X, 'conv6', config.conv_base_filter * 8, config.conv_base_filter * 8,train))
        X = tf.nn.relu(conv2d(X, 'conv7', config.conv_base_filter * 8, config.conv_base_filter * 8,train))
        return X

#

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid, forget_bias=1.0):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name
        self._forget_bias = forget_bias

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, inputs, state, scope=None):
        c_tm1, h_tm1 = tf.split(axis=1,num_or_size_splits=2,value=state)

        h_i_input=tf.concat(axis=1, values=[inputs, h_tm1])
        h_i_input = tf.reshape(h_i_input,shape=[-1,self._n_in + self._n_hid])
        w = create_weight(name='encoder_LSTM_W',shape=[(self._n_in + self._n_hid),4 * self._n_hid])
        b = create_bias(name='encoder_LSTM_B',shape=[4 * self._n_hid])
        gates = tf.matmul(h_i_input, w)

        gates = tf.nn.bias_add(gates, b)

        i_t,f_t,o_t,g_t = tf.split(axis=1, num_or_size_splits=4, value=gates)

        c_t = tf.nn.sigmoid(f_t+self._forget_bias)*c_tm1 + tf.nn.sigmoid(i_t)*tf.tanh(g_t)
        h_t = tf.nn.sigmoid(o_t)*tf.tanh(c_t)

        new_state = tf.concat(axis=1, values=[c_t,h_t])

        return h_t,new_state


def encoder_bi_LSTM(img,name):
    with tf.name_scope(name):
        img=tf.transpose(img,[0,2,1,3])
        shape = tf.shape(img)
        batch_size, H, W, C = shape[0], shape[1], shape[2], shape[3]
        img = tf.reshape(img, [batch_size * H, W, C])
        img.set_shape([None, None, config.vgg_output_deep])

        encoder_init_fw = tf.get_variable(name + 'encoder_init_fw',
                          initializer=tf.zeros((1,2 *config.encoder_rnn_size),dtype='float32'))
        encoder_init_fw =tf.tile(encoder_init_fw, [batch_size*H, 1])

        encoder_init_bw = tf.get_variable(name + 'encoder_init_bw',
                          initializer=tf.zeros((1,2 *config.encoder_rnn_size) ,dtype='float32'))
        encoder_init_bw = tf.tile(encoder_init_bw, [batch_size*H, 1])

        cell_fw = LSTMCell(name+'_fw',config.vgg_output_deep,config.encoder_rnn_size)

        cell_bw = LSTMCell(name+'_bw',config.vgg_output_deep,config.encoder_rnn_size)

        enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,inputs=img,
                                          initial_state_fw=encoder_init_fw,
                                          initial_state_bw=encoder_init_bw,
                                          dtype=tf.float32)
        # lstm_fw_cell = tf.contrib.rnn.LSTMCell(config.encoder_rnn_size, state_is_tuple=True)
        # lstm_bw_cell = tf.contrib.rnn.LSTMCell(config.encoder_rnn_size, state_is_tuple=True)
        # enc_output, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, img, dtype=tf.float32)
        enc_output=tf.concat([enc_output[0], enc_output[1]], axis=2)
        enc_output=tf.reshape(enc_output,[batch_size,H, W,2*config.encoder_rnn_size])
        enc_output = tf.transpose(enc_output,[0,2,1,3])
        enc_output = tf.reshape(enc_output, [batch_size,-1, 2 * config.encoder_rnn_size])
        return enc_output

def embedding( name,indices):
    with tf.name_scope(name):
        embedding=tf.get_variable(name='embedding',initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  shape=[config.n_symbols,config.emb_dim])
        return tf.nn.embedding_lookup(embedding, indices)


#     v_hat=b*T*(2*encoder_rnn_size)
 #    ht=b*decoder_rnn_size
class  Attetion_Rnn(tf.nn.rnn_cell.RNNCell):
    def __init__(self,V_hat):
        self.V_hat=V_hat

    @property
    def state_size(self):
        return config.decoder_rnn_size

    @property
    def output_size(self):
        return config.decoder_rnn_size

    def __call__(self, _input, state,scope=None):
        p_ht,p_c,p_out=tf.split(axis=1,num_or_size_splits=3,value=state) #c-> cell

        c_input = tf.concat([_input, p_out],axis=1) #c ->current input
        w = create_weight(name='decoder_LSTM_W',shape=[(config.emb_dim + config.decoder_rnn_size),
                                   4 * config.decoder_rnn_size])
        b = create_bias(name='decoder_LSTM_B', shape=[4 * config.decoder_rnn_size])
        gates = tf.matmul(c_input, w)
        gates = tf.nn.bias_add(gates, b)
        i_t, f_t, o_t, g_t = tf.split(axis=1, num_or_size_splits=4, value=gates)
        c_t = tf.nn.sigmoid(f_t) * p_c + tf.nn.sigmoid(i_t) * tf.tanh(g_t)
        ht = tf.nn.sigmoid(o_t) * tf.tanh(c_t)

        liner_w=create_weight(name='liner_w',shape=[config.decoder_rnn_size,2*config.encoder_rnn_size])
        liner_b=create_bias(name='liner_b', shape=[2*config.encoder_rnn_size])
        h_t=tf.matmul(ht,liner_w,name='liner_change')
        h_t=tf.nn.bias_add(h_t,liner_b)
        # h_t b*
        h_t=tf.expand_dims(h_t,axis=2)
        a_t=tf.matmul(self.V_hat,h_t,name='a_t')
        batch_size=tf.shape(a_t)[0]
        a_t=tf.reshape(a_t,(batch_size,-1))

        a_t=tf.nn.softmax(a_t)
        a_t = tf.expand_dims(a_t,2)
        z_t = tf.reduce_sum(a_t*self.V_hat,axis=1)

        s_out = tf.concat([z_t, ht], axis=1)
        out_w = create_weight(name='out_w',shape=[config.decoder_rnn_size + 2 * config.encoder_rnn_size,
                                       config.decoder_rnn_size])
        out_b = create_bias(name='out_b',  shape=[config.decoder_rnn_size])
        output = tf.nn.bias_add(tf.matmul(s_out, out_w), out_b)

        new_state = tf.concat(axis=1,values=[ht,c_t,output])

        return output,new_state


def decoder(V_hat,inputs,seq_len):
    with tf.name_scope('Decoder'):
        batch_size=tf.shape(V_hat)[0]
        p_state = tf.get_variable(name='decoder_state_0',
                        initializer=tf.zeros((1, 3 * config.decoder_rnn_size),dtype='float32'))
        decoder_state = tf.tile(p_state, [batch_size, 1])
        cell=Attetion_Rnn(V_hat)
        out,state = tf.nn.dynamic_rnn(cell, inputs, initial_state=decoder_state,
                                sequence_length=seq_len, swap_memory=True)
        return out,state







