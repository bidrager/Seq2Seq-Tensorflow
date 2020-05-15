import  model4.build_model  as build_model
import tensorflow as tf
import model4.config as config
import cv2
import os
import json
from tqdm import tqdm

# 为提供标签数据，识别结果为数字类别

import tensorflow.compat.v1 as tf #使用1版本
tf.disable_v2_behavior()

x=tf.placeholder(shape=[None,None,None,3],dtype=tf.float32)
seq_lengths=tf.placeholder(dtype=tf.int32)
labels=tf.placeholder(shape=[None,None],dtype=tf.int32)
mask=tf.placeholder(shape=[None,None,None],dtype=tf.float32)

test_dir='image/'
test_files=os.listdir(test_dir)

# with open('../process_data/id2word.json','r') as f:
#     id2word=json.load(f)

word_pre_result=dict()

def predict(sess):
    prediction=build_model.build_model(x,labels,seq_lengths,False,1)
    output = tf.argmax(tf.nn.softmax(prediction, axis=2), axis=2)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(config.saved_model)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print('successed')
    for image_file in tqdm(test_files):
        image_path = os.path.join(test_dir, image_file)
        image = cv2.imread(image_path)
        if image.shape[0]>16:
            image = (image - 127.5) / 127.5
            test_label = [[1, 0]]
            indice=1
            while indice<36 and test_label[0][-2]!=0:
                lengths = [indice]
                pre = sess.run(output, feed_dict={x: [image],labels: test_label, seq_lengths: lengths})
                test_label[0].insert(-1, pre[0][-1])
                indice+=1

            result=test_label[0][1:-2]
            res=[]
            for ind in result:
                res.append(int(ind))
            word_pre_result[image_file]=res
    with open('../result/result.json','w') as f:
        json.dump(word_pre_result,f)




if __name__=='__main__':

    sess = tf.Session()
    predict(sess)

