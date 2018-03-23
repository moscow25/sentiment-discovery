import numpy as np
#import pandas as pd
import csv, os, time, re, json, pickle

"""
Just input positive and negative examples. Options to class balance, shuffle, etc.

Assume that examples are pickled.

"""

# Remove artifacts -- mostly badly rendered emoji
def cleanup_text(t, debug = False):
    # First attempt at unreadable characters
    t_re = re.sub(r"\\x[0-9a-z]{2}", "", t)
    # Remove URLs
    t_re = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', t)
    # Second attempt at unreadable characters
    t_re = "".join(x for x in t_re if (x.isspace() or 31 < ord(x) < 127))
    # Bad parsing around ',' and '?'
    t_re = t_re.replace(' ,', ',')
    # Bad parsing around ','
    t_re = t_re.replace(' ?', '?')
    # Remove extra spaces
    t_re = ' '.join(t_re.split())
    if t != t_re and debug:
        print('%s --> %s' % (t, t_re))
    return t_re

# drop items that don't fit -- default to something that could be a tweet
MIN_LENGTH = 60
MAX_LENGTH = 280
def filter_length(arr, min_len = MIN_LENGTH, max_len = MAX_LENGTH):
    bad_indices = []
    for i in range(len(arr)):
        if len(arr[i]) < min_len or len(arr[i]) > max_len:
            bad_indices.append(i)
    return np.delete(arr, bad_indices)

# in pickle format
#positives_filenames = ["JennyJohnsonHi5.pickle", "meganamram.pickle", "anthonyjeselnik.pickle", "FunnyAsianDude.pickle"] #["humorous_oneliners.pickle"]
#negatives_filenames = ["nvidia_tweets.pickle", "elonmusk.pickle"] #, "wiki_sentences.pickle"]
positives_filenames = ["GSElevator.pickle", "SteveMartinToGo.pickle", "SICKOFWOLVES.pickle", "shitmydadsays.pickle", "JennyJohnsonHi5.pickle", "meganamram.pickle", "anthonyjeselnik.pickle", "FunnyAsianDude.pickle"]
negatives_filenames = ["nvidia_tweets.pickle", "elonmusk.pickle", "cnn.pickle", "espn.pickle", "CARandDRIVER.pickle", "fredwilson.pickle", "SenTedCruz.pickle"]

# Import the files into memory
pos_texts = []
neg_texts = []

for fn in positives_filenames:
    print('loading *pos* examples from %s' % fn)
    fin = pickle.load(open(fn, 'rb'))
    print('\t-> %d examples [%s]' % (len(fin), fin[:4]))
    pos_texts += fin
for fn in negatives_filenames:
    print('loading *neg* examples from %s' % fn)
    fin = pickle.load(open(fn, 'rb'))
    print('\t-> %d examples [%s]' % (len(fin), fin[:4]))
    neg_texts += fin

pos_texts = np.array(pos_texts)
neg_texts = np.array(neg_texts)

print('Loaded %d pos and %d neg examples before filtering.' % (len(pos_texts), len(neg_texts)))


# Perform filtering on both.
# Do we remove or down-sample short text [for non jokes?]
# Do we remove or down-sample Q&A jokes? Those that involve a question.

# Clean non-ASCII since we don't do well with these
pos_texts = [cleanup_text(t) for t in pos_texts]
neg_texts = [cleanup_text(t) for t in neg_texts]

pos_texts = filter_length(pos_texts, min_len = MIN_LENGTH, max_len = MAX_LENGTH)
neg_texts = filter_length(neg_texts, min_len = MIN_LENGTH, max_len = MAX_LENGTH)
print('Loaded %d pos and %d neg examples after length filtering [%d - %d].' % (len(pos_texts), len(neg_texts),
    MIN_LENGTH, MAX_LENGTH))

# TODO: Show distribution of lengths! Do they match, more or less??

# Shuffle both datasets
np.random.shuffle(pos_texts)
np.random.shuffle(neg_texts)


# Take the min and class balance
min_len = min(len(pos_texts), len(neg_texts))
pos_texts = pos_texts[:min_len]
neg_texts = neg_texts[:min_len]


print('Loaded %d pos and %d neg examples before saving.' % (len(pos_texts), len(neg_texts)))


text = np.concatenate((pos_texts, neg_texts))
lab = np.concatenate((np.ones(len(pos_texts)), np.zeros(len(neg_texts))))
shuf = np.random.permutation(len(text))
text = text[shuf]
lab = lab[shuf]
print(text.size)
print(lab.size)

# Dump to train/val/test
VAL_PERCENT = 0.2
TEST_PERCENT = 0.2
test_set=TEST_PERCENT
val_set=VAL_PERCENT
cat_prefix='1L'
category='tweet_comedy12_v_nv_musk_news_cars_wilson_cruz_tweets'

vs_all_string = ''
basepath = 'jokes'
fileout_path = basepath+'/'+cat_prefix+'_'+category.lower()+'/'+cat_prefix+'_'+category.lower()+vs_all_string+'_data.csv'
fileout_path_test = basepath+'/'+cat_prefix+'_'+category.lower()+'/'+cat_prefix+'_'+category.lower()+vs_all_string+'_data_test.csv'
fileout_path_val = basepath+'/'+cat_prefix+'_'+category.lower()+'/'+cat_prefix+'_'+category.lower()+vs_all_string+'_data_val.csv'

print('\n---------\nwriting to %s' % fileout_path)
if test_set > 0.:
    print('test data to %s' % fileout_path_test)
if val_set > 0.:
    print('val data to %s' % fileout_path_val)
    val_set = val_set / (1 - test_set)

try:
    os.makedirs(os.path.dirname(fileout_path))
except OSError as exc:
    print('dir already exists')

with open(fileout_path,'w') as f:
    with open(fileout_path_test, 'w') as f_test:
        with open(fileout_path_val, 'w') as f_val:
            if val_set > 0.:
                c_val = csv.writer(f_val)
                c_val.writerow(['sentence', 'label'])
            if test_set > 0.:
                c_test = csv.writer(f_test)
                c_test.writerow(['sentence', 'label'])
            c = csv.writer(f)
            c.writerow(['sentence', 'label'])

            for row in zip([s.encode("utf-8") for s in text],lab):
                if np.random.random() < test_set:
                    c_test.writerow(row)
                elif np.random.random() < val_set:
                    c_val.writerow(row)
                else:
                    c.writerow(row)