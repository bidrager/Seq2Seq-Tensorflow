from model4.ops import *
import model4.config as cfg
import cv2
import tensorflow as tf
import json
import os
import time
import model4.dense_net

import tensorflow.compat.v1 as tf #使用1版本
tf.disable_v2_behavior()

# 这里是预先处理好的训练数据（本程序未提供数据），运行use_predict.py这个地方报错就先注释了训练是需要打开的，下面的注释同理

# with open(cfg.label_file,'r') as f:
#     all_data=json.load(f)

# batch_size=cfg.batch_size
#
# all_images=[]
# all_image_dir=cfg.all_image_dir
# image_dirs=os.listdir(all_image_dir)
# image_dirs_dict=dict()
# image_dirs_index=dict()
# for dir in image_dirs:
#     image_dirs_index[dir]=0
#     single_dir_images=os.listdir(os.path.join(all_image_dir, dir))
#     np.random.shuffle(single_dir_images)
#     image_dirs_dict[dir]=single_dir_images

def padding_image(image,bucket):
    height,width,channel=image.shape
    image = (image - 127.5)/127.5
    new_image=np.zeros(([bucket[1],bucket[0],3]))
    h=int((bucket[1]-height)/2)
    w=int((bucket[0]-width)/2)
    new_image[h:height+h,w:width+w,:]=image
    return new_image

# def get_train_data(batch_size):
#     image_dir=np.random.choice(image_dirs,p=cfg.P)
#     image_path = os.path.join(all_image_dir, image_dir)
#     if image_dirs_index[image_dir]+batch_size<len(image_dirs_dict[image_dir]):
#         target_images=image_dirs_dict[image_dir][image_dirs_index[image_dir]:image_dirs_index[image_dir]+batch_size]
#         image_dirs_index[image_dir]=image_dirs_index[image_dir]+batch_size
#     else:
#         target_images=image_dirs_dict[image_dir][image_dirs_index[image_dir]:len(image_dirs_dict[image_dir])]
#         image_dirs_index[image_dir] = (image_dirs_index[image_dir] + batch_size)%len(image_dirs_dict[image_dir])
#         np.random.shuffle(image_dirs_dict[image_dir])
#         target_images=target_images+image_dirs_dict[image_dir][0:image_dirs_index[image_dir]]
#     data=[]
#     labels=[]
#     lengths=[]
#     hs=[]
#     ws=[]
#     for image in target_images:
#         image_name=os.path.join(image_path,image)
#         image_data=cv2.imread(image_name)
#         h,w,c=image_data.shape
#         hs.append(h)
#         ws.append(w)
#         data.append(image_data)
#         labels.append(all_data[image])
#         lengths.append(len(labels[-1]))
#     h=np.max(hs)
#     w=np.max(ws)
#     image_datas=[]
#     for image in data:
#         image_data=padding_image(image,[w,h])
#         image_datas.append(image_data)
#     return image_datas,labels,lengths,image_dir


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)

            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def build_model(images,labels,label_seq_len=None,istraining=True,single_batch=cfg.single_batch):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE) as scope:
        img_feature_map=resNet(images,istraining)
        encoder_output=encoder_bi_LSTM(img_feature_map,name='encoder')
        decoder_inputs=embedding('embedding',labels[:,:-1])
        decoder_out,decoder_state=decoder(encoder_output,decoder_inputs,label_seq_len)
        decoder_out=tf.reshape(decoder_out,[-1,cfg.decoder_rnn_size])
        softmax_w = create_weight(name='softmax_w', shape=[cfg.decoder_rnn_size, cfg.n_symbols])
        softmax_b = create_bias(name='softmax_b', shape=[cfg.n_symbols])
        logits = tf.nn.bias_add(tf.matmul(decoder_out,softmax_w),softmax_b)
        logits=tf.reshape(logits,[single_batch,-1,cfg.n_symbols])
        return logits


def train(sess):
    x = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)
    seq_lengths = tf.placeholder(dtype=tf.int32)
    labels = tf.placeholder(shape=[None, None], dtype=tf.int32)
    mask = tf.placeholder(shape=[None, None, None], dtype=tf.float32)

    m_x = tf.split(axis=0, num_or_size_splits= len(cfg.num_gpu), value=x)
    m_seq_lengths = tf.split(axis=0, num_or_size_splits=len(cfg.num_gpu), value=seq_lengths)
    m_labels = tf.split(axis=0, num_or_size_splits=len(cfg.num_gpu), value=labels)
    m_mask = tf.split(axis=0, num_or_size_splits=len(cfg.num_gpu), value=mask)

    grad_all = []
    losses=[]
    optimizer = tf.train.AdamOptimizer(cfg.learning_rate)
    outputs=[]
    for i in range(len(cfg.num_gpu)):
        with tf.device('/gpu:%d' % cfg.num_gpu[i]):
            prediction=build_model(m_x[i],m_labels[i],m_seq_lengths[i])
            output = tf.argmax(tf.nn.softmax(prediction, axis=2), axis=2)
            s_labels = tf.one_hot(m_labels[i], cfg.n_symbols, 1, 0)
            s_labels = tf.cast(s_labels, dtype=tf.float32)
            cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,
                                                                labels=s_labels[:, 1:, :])
            loss = tf.reduce_sum(cross_entropy_loss * m_mask[i]) / tf.reduce_sum(m_mask[i]) * cfg.n_symbols
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            grad_all.append(capped_gvs)
            losses.append(loss)
            outputs.append(output)
    loss=tf.reduce_mean(losses)
    ave_grad = average_gradients(grad_all)
    train_step = optimizer.apply_gradients(ave_grad)
    outputs=tf.reshape(outputs,[cfg.batch_size,-1])
    init=tf.global_variables_initializer()
    sess.run(init)
    saver=tf.train.Saver(max_to_keep=5)
    checkpoint = tf.train.get_checkpoint_state(cfg.saved_model)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
    for step in range(cfg.start_step,cfg.train_steps):
        images, labels_batch, lengths, image_dir_name = get_train_data(batch_size)
        max_length = np.max(lengths)
        mask_batch = np.zeros([cfg.batch_size, max_length, cfg.n_symbols], dtype=np.float32)
        max_length = max_length + 1
        for i in range(len(lengths)):
            mask_batch[i, 0:lengths[i], :] = 1.0
            pad_label = [0] * (max_length - lengths[i])
            labels_batch[i] = labels_batch[i] + pad_label
        sess.run(train_step,feed_dict={x: images,
                                labels: labels_batch, seq_lengths: lengths,mask:mask_batch})
        if step%100==0:
            los=sess.run(loss, feed_dict={x: images,
                                            labels: labels_batch, seq_lengths: lengths,mask: mask_batch})
            print(image_dir_name,los)
        if step%cfg.save_step==0 and step>1:
            cfg.learning_rate=cfg.learning_rate*0.9
            saver.save(sess, cfg.saved_model+'/recognotion', global_step=step)
            t=time.time()
            images, labels_batch, _, image_dir_name = get_train_data(cfg.batch_size)
            test_label=[[1,0] for j in range(cfg.batch_size)]
            for  indice in range(1,36):
                lengths=[indice]*cfg.batch_size
                pre=sess.run(outputs,feed_dict={x: images,labels:test_label,seq_lengths: lengths})
                for i in range(cfg.batch_size):
                    test_label[i].insert(-1,pre[i][-1])
            t1=time.time()
            acc=0
            word_acc=0
            wordsum=0
            for i in range(cfg.batch_size):
                test=test_label[i][1:-1]
                print('predict: ', test)
                label = labels_batch[i][1:]
                print('label: ', label)
                if len(label)<36:
                    cur_flag=0
                    for index in range(len(label)):
                        if label[index]==test[index]:
                            word_acc+=1
                            cur_flag+=1
                        wordsum+=1
                    if cur_flag==len(label):
                        acc+=1
            print('accuracy: ',acc/cfg.batch_size)
            print('word_accuracy: ', word_acc / wordsum)
            print('test_time: ',t1-t,image_dir_name)




if __name__=='__main__':
    import os

    config = tf.ConfigProto(allow_soft_placement=True)
    sess=tf.Session(config=config)
    train(sess)





