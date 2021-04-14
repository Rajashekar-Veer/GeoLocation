###preprocessing options###
word_minfreq=5
seq_minlen=5
# bucket_sizes=[50,100,150,200,250,300]
bucket_sizes=[50]


###hyper-parameters###
#general
batch_size=512
epoch_size=10
keep_prob=0.8
seed=1
learning_rate=0.001

# penultimate representation layer size
# R size => 100, 200, 300, 400
rep_hidden_size=400

# gaussian standard deviation of noise
# 0.1 => with noise
# 0.0 => without noise
corruption_level=0.0

# sparsity loss factor
# 0.1 => with loss
# 0.0 => without loss
alpha=0.0

#text feature
# text_emb_size=200 #same as rnn hidden size
text_emb_size=300 #same as rnn hidden size
text_filter_number=400
text_pool_window =10 #max pool window size
#time feature (0 to turn off; ditto for others)
time_size=50
#day feature
day_size=0
#utc offset feature
offset_size=0
#timezone feature
timezone_size=0
#location feature
loc_maxlen=20
loc_emb_size=300
loc_filter_width=3
loc_filter_number=300 #(0 to turn off)
#user description feature
desc_maxlen=0
desc_emb_size=0
desc_filter_width=0
desc_filter_number=0 #(0 to turn off)
#name feature
name_maxlen=0
name_emb_size=0
name_filter_width=0
name_filter_number=0 #(0 to turn off)
#user created time feature
usertime_size=10

###misc###
verbose=True
save_model=True

###word2vec flag###
use_word2vec=True

###input/output###
output_dir="Deepgeo/models/"
#for a quick test of the code, you can train on "valid" data and test on "valid" data, e.g.
train_data="Deepgeo/data/training_data/training.json.data1.json"
train_label="Deepgeo/data/training_data/training.json.label1.json"
# train_data="Deepgeo/input/valid/data.tweet.json"
# train_label="Deepgeo/input/valid/label.tweet.json"
valid_data="Deepgeo/input/valid/data.tweet.json"
valid_label="Deepgeo/input/valid/label.tweet.json"
output_prefix="default"
# output_prefix="wmin%d_smin%d_bucket%s_batch%d_epoch%d_kp%.1f_seed%d_lr%.4f_h%d_cl%.3f_a%.3f_text%d-%d-%d_time%d_day%d_offset%d_tz%d_loc%d-%d-%d-%d_desc%d-%d-%d-%d_name%d-%d-%d-%d_utime%d" % \
#     (word_minfreq, seq_minlen, ("-".join([str(item) for item in bucket_sizes])), batch_size, epoch_size,
#     keep_prob, seed, learning_rate, rep_hidden_size, corruption_level, alpha,
#     text_emb_size, text_filter_number, text_pool_window,
#     time_size, day_size, offset_size, timezone_size,
#     loc_maxlen, loc_emb_size, loc_filter_width, loc_filter_number,
#     desc_maxlen, desc_emb_size, desc_filter_width, desc_filter_number,
#     name_maxlen, name_emb_size, name_filter_width, name_filter_number, usertime_size)
