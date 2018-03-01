import numpy as np
import pandas as pd
import csv, os

#column_name = 'best_emotion'
column_name = 'what_is_the_authors_sentiment_feeling_throughout_the_post'
column_value_name = column_name + ':confidence'

# if we care about opposite categories?
opposite_map = {'1':'3', '2':'2', '3':'1'}
save_neutrals = True # False # do we keep 0.5 labels for a category?

def get_categories(category_column):
        s = set()
        for labels in category_column:
                s |= set(labels.split())
        return list(s)

def get_category_labels(category, category_column, label_column, label_threshold = 0.5):
        rtn=[]
        opposite_category = opposite_map[category]
        for cats, labs in zip(category_column, label_column):
                print('----')
                print(cats)
                print(labs)
                cat_list = cats.split()
                lab_list = labs.split()
                print(cat_list)
                print('computing for category |%s|' % category)
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
                                print('positive case!')
                                rtn.append(1)
                                neutral = False
                elif opposite_category in cat_list:
                        # Negative label if *opposite* category above a threshold
                        cat_idx = cat_list.index(opposite_category)
                        if float(lab_list[cat_idx]) >= label_threshold:
                                print('negative case!')
                                rtn.append(0)
                                neutral = False
                # Everything else is neutral (ignore or use?)
                if neutral:
                        print('neutral for %s' % category)
                        rtn.append(0.5)
        return np.array(rtn)

def get_category_text_label_column(csv_path):
        c = pd.read_csv(csv_path)
        return c[column_name].values.tolist(), c['text'].values.tolist(), c[column_value_name].values.tolist()

def write_csv(basepath, category, text, lab, save_neutrals=True):
        fileout_path = basepath+'/'+category.lower()+'/data.csv'
        print('writing to %s' % fileout_path)
        try:
                os.makedirs(os.path.dirname(fileout_path))
        except OSError as exc:
                print('dir already exists')
        with open(fileout_path,'w') as f:
                c = csv.writer(f)
                c.writerow(['sentence', 'label'])
                print('average label value %.3f' % np.mean(lab))
                for row in zip([s.encode("utf-8") for s in text],lab):
                        if (not save_neutrals) and row[1] == 0.5:
                                continue
                        c.writerow(row)

#input_filename = 'crowdflower.plutchik.csv'
input_filename = 'a1204167-single-label-sentiment.csv'

cats, text, lab = get_category_text_label_column(input_filename)
# Examples
for i in range(20):
        print('----------')
        print('%s\t%s\t%s' % (cats[i], text[i], lab[i]))
basepath = 'crowdflower'
categories = get_categories(cats)
for c in categories:
        l = get_category_labels(c, cats, lab)
        write_csv(basepath, c, text, l, save_neutrals = save_neutrals)