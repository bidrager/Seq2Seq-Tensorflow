import tensorflow as tf
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
    return w
def create_bias(shape,name):
    b =tf.get_variable(initializer=tf.constant_initializer(value=0,dtype=tf.float32),
                           name='bias_'+name,shape=shape)
    return b

def conv2d(input,name,filter_shape,stride,train):
    with tf.name_scope(name):
        conv1_w=create_weight(shape=filter_shape,name=name+'_w')
        conv1_b=create_bias(filter_shape[-1],name=name+'_b')
        conv1=tf.nn.conv2d(input,conv1_w,strides=[1,stride,stride,1],padding='SAME')
        conv=tf.nn.bias_add(conv1,conv1_b)
        conv=norm_layer(name+'_batch_norm',conv,train)
        return conv

def dense_block(x, nb_layers, layer_name,train):
    with tf.name_scope(layer_name):
        for i in range(nb_layers):
            previous_deep = x.get_shape().as_list()[-1]
            x_ = conv2d(x,name=layer_name + str(i),
                        filter_shape=[3,3,previous_deep,config.growth_deep],stride=1,train=train)
            x_=tf.nn.relu(x_)
            x=tf.concat([x,x_],axis=3)
        return x

def transition_layer( x,deep,name,train):
    with tf.name_scope(name):
        previous_deep = x.get_shape().as_list()[-1]
        x = conv2d(x, filter_shape=[3,3,previous_deep,deep], stride=1, name=name+'transition_conv',train=train)
        x=tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return x

def dense_net(input,train):
    with tf.name_scope('dense_net'):
        x = conv2d(input, filter_shape=[5,5,3,2 * config.conv_base_filter], stride=1, name='conv0',train=train)
        x=tf.nn.relu(x)

        x=dense_block(x,2,'dense_block_0',train=train)
        x=transition_layer(x,4* config.conv_base_filter,name='transition_layer_1',train=train)

        x = dense_block(x, 2, 'dense_block_1', train=train)
        x = transition_layer(x, 6 * config.conv_base_filter, name='transition_layer_2', train=train)

        x = dense_block(x, 2, 'dense_block_2', train=train)
        x = transition_layer(x, 8 * config.conv_base_filter, name='transition_layer_3', train=train)

        x = dense_block(x, 2, 'dense_block_3', train=train)
        x = transition_layer(x, 10 * config.conv_base_filter, name='transition_layer_0', train=train)

        x = dense_block(x, 2, 'dense_block_4', train=train)
        return x




