import calendar
import codecs
import ujson
import sys
import operator
import numpy as np
from collections import defaultdict
import os
from nltk.tokenize import RegexpTokenizer
from SMART import word2vec_path
import gensim

# globals
day_map = {v: i for i, v in enumerate(list(calendar.day_abbr))}  # {"Mon": 0, ...}
unk_token = "<unk>"
pad_token = "<pad>"
id_city_key = {"hashed_tweet_id": ("hashed_tweet_id", "city"), "tweet_id": ("tweet_id", "tweet_city"), \
               "id_str": ("id_str", "tweet_city")}

word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
# word2vec = []


def datetime_to_float(datetime):
    times = datetime.split()[3].split(":")
    t = ((float(times[0]) * 60 + float(times[1])) % 1440) / float(1440)
    return t


def get_id_city_key(d):
    id_key, city_key = None, None
    for k in list(id_city_key.keys()):
        if k in d:
            id_key = id_city_key[k][0]
            city_key = id_city_key[k][1]
    if id_key == None or city_key == None:
        print("Unable to find tweet ID and city key; json =", d)
        raise SystemExit

    return id_key, city_key


def load_label(filename, cf):
    label, city_coordinates, tweet_coordinates = {}, {}, {}
    id_key, city_key = None, None

    for line_id, line in enumerate(codecs.open(filename, "r", "utf-8")):
        d = ujson.loads(line)
        if id_key == None:
            id_key, city_key = get_id_city_key(d)
        label[d[id_key]] = d[city_key]

        # dict from city label string to city coordiantes
        if 'user_city' in d and d[city_key] not in city_coordinates:
            city_coordinates[d[city_key]] = (d['user_city_latitude'], d['user_city_longitude'])

        # dict from city label string to the first encountered tweet coordinate
        if 'tweet_city' in d and d[city_key] not in tweet_coordinates:
            tweet_coordinates[d[city_key]] = (d['tweet_latitude'], d['tweet_longitude'])

        if line_id % 10000 == 0 and cf.verbose:
            sys.stdout.write(str(line_id) + " lines processed\r")
    return label, city_coordinates, tweet_coordinates


def load_data(filename, label, check_min_len, cf):
    data = []
    id_key = None
    for line_id, line in enumerate(codecs.open(filename, "r", "utf-8")):
        d = ujson.loads(line)
        if id_key == None:
            id_key, _ = get_id_city_key(d)
        if (d[id_key] in label) and ((not check_min_len) or (len(d["text"]) >= cf.seq_minlen)):
            tweet_dic = build_tweet_dic(d, id_key)
            data.append(tweet_dic)

        if line_id % 10000 == 0 and cf.verbose:
            sys.stdout.write(str(line_id) + " lines processed\r")
    return data


def build_tweet_dic(data, id_key):
    d = {}
    d["text"] = data["text"]
    d["id_str"] = data[id_key]

    # time is converted to number of minutes since 00:00 and then normalised to 0-1
    d["time"] = datetime_to_float(data["created_at"])

    # mon = 0, tue = 1, etc
    d["day"] = day_map[data["created_at"].split()[0]]

    # utc offset (user-level metadata)
    d["offset"] = 100.0

    # time zone (user-level metadata)
    d["timezone"] = data["user"]["time_zone"]
    if d["timezone"] == None:
        d["timezone"] = unk_token

    # location (user-level metadata)
    d["location"] = data["user"]["location"]

    # description (user-level metadata)
    d["description"] = data["user"]["description"]

    # name (user-level metadata)
    d["name"] = data["user"]["name"]

    # user created time (user-level metadata)
    d["usertime"] = datetime_to_float(data["user"]["created_at"])

    return d


def get_vocab(data, attr_name, granularity, minfreq):
    wordfreq = defaultdict(int)
    dic = {unk_token: 0, pad_token: 1}
    dic_rev = [unk_token, pad_token]
    for d in data:
        if granularity == "char":
            for c in d[attr_name]:
                wordfreq[c] += 1
        elif granularity == "word":
            wordfreq[d[attr_name]] += 1

    for w, f in sorted(list(wordfreq.items()), key=operator.itemgetter(1), reverse=True):
        if f >= minfreq:
            if (w != unk_token) and (w != pad_token):
                dic[w] = len(dic_rev)
                dic_rev.append(w)
        else:
            break

    return dic, dic_rev, wordfreq


def get_classes(data, label, city_coordinates, tweet_coordinates):
    classes, coordinates = {unk_token: 0}, [(0.0, 0.0)]  # for missing classes in test
    miss_count = 0

    for d in data:
        c = label[d["id_str"]]
        if c not in classes:
            try:
                coordinate = city_coordinates[c]
            except KeyError:
                coordinate = tweet_coordinates[c]
                miss_count += 1

            coordinates.append(coordinate)
            classes[c] = len(classes)

    print(f'Number of cities mapped to tweet coordinate: {miss_count}')
    return classes, coordinates


def convert_text_to_id(text, vocab_dic):
    x = []
    if text != None:
        for c in text:
            if c in vocab_dic:
                x.append(vocab_dic[c])
            else:
                x.append(vocab_dic[unk_token])
    return x


def clean_data(data, label, vocabxid, tzxid, locxid, descxid, namexid, classes, cf):
    len_x, miss_y, len_loc, len_desc, len_name = [], 0, [], [], []
    text_maxlen = max(cf.bucket_sizes)
    for di, d in enumerate(data):
        # label
        c = label[d["id_str"]]
        if c in classes:
            d["y"] = classes[c]
        else:
            d["y"] = classes[unk_token]
            miss_y += 1

        # tweet text
        x = convert_text_to_id(d["text"], vocabxid)
        len_x.append(len(x))

        # user time zone
        if d["timezone"] in tzxid:
            d["timezone"] = tzxid[d["timezone"]]
        else:
            d["timezone"] = tzxid[unk_token]
        d["x"] = x[:text_maxlen]

        # location
        loc_text = convert_text_to_id(d["location"], locxid)
        d["location"] = loc_text[:cf.loc_maxlen]
        len_loc.append(len(loc_text))

        # description
        desc_text = convert_text_to_id(d["description"], descxid)
        d["description"] = desc_text[:cf.desc_maxlen]
        len_desc.append(len(desc_text))

        # name
        name_text = convert_text_to_id(d["name"], namexid)
        d["name"] = name_text[:cf.name_maxlen]
        len_name.append(len(name_text))

        if di % 10000 == 0 and cf.verbose:
            sys.stdout.write(str(di) + " instances processed\r")

    return len_x, miss_y, len_loc, len_desc, len_name


def process_data(data, vocabxid, tzxid, locxid, descxid, namexid, classes, cf):
    text_maxlen = max(cf.bucket_sizes)
    status_dics = []

    for d in data:
        status = {}
        status['text'] = d['text']
        status['id_str'] = str(d['tweetID'])

        # label
        status['y'] = classes[unk_token]

        # tweet text
        x = convert_text_to_id(d['text'], vocabxid)
        status['x'] = x[:text_maxlen]

        # time is converted to number of minutes since 00:00 and then normalised to 0-1
        status['time'] = datetime_to_float(d['date'])

        # mon = 0, tue = 1, etc
        status['day'] = day_map[d['date'].split()[0]]

        # utc offset (user-level metadata)
        status['offset'] = 100.0

        # user created time (user-level metadata)
        status['usertime'] = datetime_to_float(d['userDate'])

        # user time zone (user-level metadata)
        status['timezone'] = tzxid[unk_token]

        # location (user-level metadata)
        loc_text = convert_text_to_id(d['userLocation'], locxid)
        status['location'] = loc_text[:cf.loc_maxlen]

        # description (user-level metadata)
        desc_text = convert_text_to_id(d['userDescription'], descxid)
        status['description'] = desc_text[:cf.desc_maxlen]

        # name (user-level metadata)
        name_text = convert_text_to_id(d['userName'], namexid)
        status['name'] = name_text[:cf.name_maxlen]

        status_dics.append(status)

    return status_dics

#########################
#  PURDUE ADDED METHOD  #
#########################
def get_embedding(data, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    # print('-----')
    # print(data[0][0])
    # print('-----')
    # word2vec = []

    tokenizer = RegexpTokenizer(r'\w+')
    # data_embed = np.zeros((data.shape[0], MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
    data_embed = np.zeros((len(data), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
    # print(data_embed.shape)
    # for i in range(data.shape[0]):
    for i in range(len(data)):
        tokens = tokenizer.tokenize(data[i])    
        for j in range(len(tokens)):
            data_embed[i][MAX_SEQUENCE_LENGTH-j-1] = word2vec[tokens[j]][:EMBEDDING_DIM] if tokens[j] in word2vec else np.random.rand(EMBEDDING_DIM)
    return data_embed   


def get_batch(data, i, blen, is_training, cf):
    x, y, time, day, offset, timezone, loc, desc, name, usertime = [], [], [], [], [], [], [], [], [], []
    noise_size = cf.text_filter_number + cf.time_size + cf.day_size + cf.offset_size + \
                 cf.timezone_size + cf.loc_filter_number + cf.desc_filter_number + \
                 cf.name_filter_number + cf.usertime_size

    # get batch max len and bucket id if blen is a dictionary
    if type(blen) is int:
        bucket_len = blen
        bucket_id = -1
    # else it is an int, use that parameter as length
    else:
        l = max([len(d["x"]) for d in data[(i * cf.batch_size):((i + 1) * cf.batch_size)]])
        bucket_id, bucket_len = blen[l]
    vals = []
    for d in data[(i * cf.batch_size):((i + 1) * cf.batch_size)]:
        x.append(pad(d["x"][:bucket_len], bucket_len))
        vals.append(d['text'])
        y.append(d["y"])
        time.append(d["time"])
        day.append(d["day"])
        offset.append(d["offset"])
        timezone.append(d["timezone"])
        loc.append(pad(d["location"], cf.loc_maxlen))
        desc.append(pad(d["description"], cf.desc_maxlen))
        name.append(pad(d["name"], cf.name_maxlen))
        usertime.append(d["usertime"])

    # number of examples
    num_examples = len(x)
    # pad the batch if num_examples < batch_size
    for _ in range(cf.batch_size - len(x)):
        vals.append('')
        x.append(pad([], bucket_len))
        y.append(0)
        time.append(0.0)
        day.append(0)
        offset.append(0.0)
        timezone.append(0)
        loc.append(pad([], cf.loc_maxlen))
        desc.append(pad([], cf.desc_maxlen))
        name.append(pad([], cf.name_maxlen))
        usertime.append(0.0)

    # generate gausian noise
    if is_training and cf.corruption_level > 0:
        noise = np.random.normal(0.0, cf.corruption_level, (cf.batch_size, noise_size))
    else:
        noise = np.zeros((cf.batch_size, noise_size))

    text_embeddings = get_embedding(vals, bucket_len, cf.text_emb_size)
    return x, y, time, day, offset, timezone, loc, desc, name, usertime, noise, num_examples, bucket_id, text_embeddings

def pad(lst, max_len):
    # <pad> ID is always 1
    return lst + [1] * (max_len - len(lst))
