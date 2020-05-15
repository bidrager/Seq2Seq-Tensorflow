growth_deep=32
lambd=1e-5
label_file='../process_data/train_id_label_reinforce.json'
test_dir='../data/test/'
all_image_dir='../data/buckedted_images3/'
encoder_rnn_size=512
decoder_rnn_size=512*2
n_symbols=9000
emb_dim=50
saved_model='saved_model/'
batch_size=12
epoch=80
train_steps=int(180000*3*epoch/batch_size)+1
start_step=990001
save_step=int(180000*3/batch_size)
learning_rate=0.000002
conv_base_filter=32
vgg_output_deep=conv_base_filter*8

P=[0.15,0.1,0.2,0.15,0.4]

num_gpu=[0,1]

single_batch=int(batch_size/len(num_gpu))