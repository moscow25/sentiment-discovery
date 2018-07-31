import numpy as np
import pandas as pd
import csv, os, time, re

#column_name = 'best_emotion'
#column_name = 'what_is_the_authors_sentiment_feeling_throughout_the_post'
#column_name = 'what_emotions_is_the_author_feeling'
#column_name = 'which_of_the_joysadness_emotions_is_the_author_feeling'
column_name = 'user_emotions'
column_value_name = column_name + ':confidence'
#text_column_name = 'text'
text_column_name = 'title'
#watson_value_name = 'label'
watson_value_name = 'sentiment'

# if we care about opposite categories?
opposite_map = {'1':'3', '2':'2', '3':'1',
        'joy':'sadness', 'sadness':'joy', 'trust':'disgust', 'disgust':'trust',
        'anger': 'fear', 'fear':'anger', 'surprise':'anticipation', 'anticipation':'surprise',
        'no_emotion':'no_emotion', 'no_emotionneutral':'no_emotionneutral', }

#category_name = 'CFWatson'
#category_name = 'CFSemEvalWatson'
category_name = 'CFNVidiaRandomWatson'

# Facebook
#opposite_map = {
#        'no_emotion':'no_emotion',
#        'joy':'sadness', 'sadness': 'joy',
#        'surprise':'no_emotion', 'anger':'joy', 'lol':'sadness'}
save_neutrals = False # do we keep 0.5 labels for a category?
fill_in_class_balance = True # If no neutrals... convert enough to negatives?
LABEL_THRESHOLD = 0.30

# Remove artifacts -- mostly badly rendered emoji
def cleanup_text(t):
        # First attempt at unreadable characters
        t_re = re.sub(r"\\x[0-9a-z]{2}", "", t)
        # Remove URLs
        t_re = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', t)
        # Other rewrites
        t_re = re.sub(r'&amp;', '&', t_re)
        t_re = re.sub(r'&#039;', "'", t_re)
        t_re = re.sub(r'&gt;', '>', t_re)
        t_re = re.sub(r'&lt;', '<', t_re)
        # Second attempt at unreadable characters
        t_re = "".join(x for x in t_re if (x.isspace() or 31 < ord(x) < 127))
        if t != t_re:
                print('%s --> %s' % (t, t_re))
        return t_re

def get_categories(category_column):
        s = set()
        for labels in category_column:
                s |= set(labels.split())
        return list(s)

def get_category_labels(category, category_column, label_column, label_threshold = LABEL_THRESHOLD):
        rtn=[]
        opposite_category = opposite_map[category]
        for cats, labs in zip(category_column, label_column):
                #print('----')
                #print(cats)
                #print(labs)
                cat_list = cats.split()
                lab_list = labs.split()
                #print(cat_list)
                #print('computing for category |%s|' % category)
                # Basic rule -- do we get any labels for column X?
                #if category in cat_list:
                #        rtn.append(1)
                #else:
                #        rtn.append(0)
                # More custom rule -- enforce % label [2/3 agreement?]
                neutral = True
                if category in cat_list:
                        # Positive label if category above a threshold
                        cat_idx = cat_list.index(category)
                        if float(lab_list[cat_idx]) >= label_threshold:
                                #print('positive case!')
                                rtn.append(1)
                                neutral = False
                elif opposite_category in cat_list:
                        # Negative label if *opposite* category above a threshold
                        cat_idx = cat_list.index(opposite_category)
                        if float(lab_list[cat_idx]) >= label_threshold:
                                #print('negative case!')
                                rtn.append(0)
                                neutral = False
                # Everything else is neutral (ignore or use?)
                if neutral:
                        #print('neutral for %s' % category)
                        rtn.append(0.5)
        return np.array(rtn)

def get_category_text_label_column(csv_path, use_watson = True):
        c = pd.read_csv(csv_path)
        if use_watson:
                return c[column_name].values.tolist(), c[text_column_name].values.tolist(), c[column_value_name].values.tolist(), c[watson_value_name].values.tolist()
        else:
                return c[column_name].values.tolist(), c[text_column_name].values.tolist(), c[column_value_name].values.tolist()

# 'CF' prefix for CrowdFlower
# 'FB' prefix for Facebook
def write_csv(basepath, category, text, lab, watson = [],
        save_neutrals=False, vs_all=False, test_set=0.10, val_set=0.10, cat_prefix=category_name):
        if vs_all:
                vs_all_string = '_vs_all'
        else:
                vs_all_string = ''
        fileout_path = basepath+'/'+cat_prefix+'_'+category.lower()+'/'+cat_prefix+'_'+category.lower()+vs_all_string+'_data.csv'
        fileout_path_test = basepath+'/'+cat_prefix+'_'+category.lower()+'/'+cat_prefix+'_'+category.lower()+vs_all_string+'_data_test.csv'
        fileout_path_val = basepath+'/'+cat_prefix+'_'+category.lower()+'/'+cat_prefix+'_'+category.lower()+vs_all_string+'_data_val.csv'
        print('\n---------\nwriting to %s' % fileout_path)
        if test_set > 0.:
                print('test data to %s' % fileout_path_test)
        if val_set > 0.:
                print('val data to %s' % fileout_path_val)
                val_set = val_set / (1 - test_set)
        # Count distributions for this class:
        unique, counts = np.unique(lab, return_counts=True)
        label_counts = dict(zip(unique, counts))
        print(label_counts)
        if not 0.0 in label_counts.keys():
                label_counts[0.0] = 0.
        negatives_fillin = max((label_counts[1.0] - label_counts[0.0]),0)
        negatives_fillin_rate = negatives_fillin / (label_counts[0.5] + 0.01)
        print('fillin %d (%.3f) negative examples' % (negatives_fillin, negatives_fillin_rate))
        vs_all_fillin_rate = label_counts[1.0] / (label_counts[0.5] + label_counts[0.0] + 0.01)
        print('vs_all_fillin rate %.3f' % vs_all_fillin_rate)
        time.sleep(2)

        try:
                os.makedirs(os.path.dirname(fileout_path))
        except OSError as exc:
                print('dir already exists')
        with open(fileout_path,'w') as f:
                with open(fileout_path_test, 'w') as f_test:
                        with open(fileout_path_val, 'w') as f_val:
                                if val_set > 0.:
                                        c_val = csv.writer(f_val)
                                        c_val.writerow(['sentence', 'label', 'watson'])
                                if test_set > 0.:
                                        c_test = csv.writer(f_test)
                                        c_test.writerow(['sentence', 'label', 'watson'])
                                c = csv.writer(f)
                                c.writerow(['sentence', 'label', 'watson'])
                                print('average label value %.3f' % np.mean(lab))
                                for row in zip([s.encode("utf-8") for s in text],lab, watson):
                                        # alternatively, do vs-all-fillin [positives vs random]
                                        if vs_all:
                                                if row[1] == 1.0:
                                                        if np.random.random() < test_set:
                                                                c_test.writerow(row)
                                                        elif np.random.random() < val_set:
                                                                c_val.writerow(row)
                                                        else:
                                                                c.writerow(row)
                                                elif np.random.random() <= vs_all_fillin_rate:
                                                        if np.random.random() < test_set:
                                                                c_test.writerow((row[0],0.0,row[2]))
                                                        elif np.random.random() < val_set:
                                                                c_val.writerow((row[0],0.0,row[2]))
                                                        else:
                                                                c.writerow((row[0],0.0,row[2]))
                                                continue
                                        # else... save positives, negatives, and filling 0.5 if not enough negs
                                        if (not save_neutrals) and row[1] == 0.5:
                                                # Convert some neutrals to negatives for class balance
                                                if fill_in_class_balance and np.random.random() <= negatives_fillin_rate:
                                                        #print(row)
                                                        if np.random.random() < test_set:
                                                                c_test.writerow((row[0],0.0,row[2]))
                                                        elif np.random.random() < val_set:
                                                                c_val.writerow((row[0],0.0,row[2]))
                                                        else:
                                                                c.writerow((row[0],0.0,row[2]))
                                                continue
                                        if np.random.random() < test_set:
                                                c_test.writerow(row)
                                        elif np.random.random() < val_set:
                                                c_val.writerow(row)
                                        else:
                                                c.writerow(row)

#input_filename = 'crowdflower.plutchik.csv'
#input_filename = 'a1204167-single-label-sentiment.csv'
#input_filename = 'a1204979-first-cut-aggregated.csv'
#input_filename = 'a1207640-second-cut-aggregated.csv'
#input_filename = 'a1204979-first-second-cut-aggregated.csv'
#input_filename = 'a1213744-first-cut-FB-aggregated.csv'
#input_filename = 'a1207640-2500-plutchik.csv'
#input_filename = 'a1283272-Plutchik-binary-SemEval-6000.csv'
input_filename = 'a1282896-Plutchik-binary-NVdia-Logan-4000.csv'

cats, text, lab, watson = get_category_text_label_column(input_filename)
#cats = [item.lower() for item in cats]
cats = [item.lower() for item in cats]
text = [cleanup_text(t) for t in text]
# Examples
for i in range(20):
        print('----------')
        print('%s\t%s\t%s' % (cats[i], text[i], lab[i]))
basepath = 'crowdflower'
categories = get_categories(cats)
for c in categories:
        l = get_category_labels(c, cats, lab)
        write_csv(basepath, c, text, l, watson=watson, save_neutrals = save_neutrals)