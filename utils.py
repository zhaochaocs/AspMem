#!/usr/bin/env python

import re
import numpy as np

from sklearn.metrics import f1_score
from sklearn import preprocessing

import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

category_aspects = [
    ["Compartments", "Customer_service", "Handles", "Looks", "none", "Price", "Protection", "Quality", "Size_Fit"],
    ["Battery", "Comfort", "Connectivity", "Durability", "Ease_of_use", "Look", "none", "Price", "Sound"],
    ["Color", "Comfort", "Durability", "Look", "Materials", "none", "Price", "Size", "Weather_resistance"],
    ["Build_Quality", "Connectivity", "Extra_functionality", "Feel_Comfort", "Layout", "Looks", "Noise", "none",
     "Price"],
    ["Apps_Interface", "Connectivity", "Customer_service", "Ease_of_use", "Image", "none", "Price", "Size_Look",
     "Sound"],
    ["Accessories", "Build_Quality", "Customer_service", "Ease_of_use", "Noise", "none", "Price", "Suction_Power",
     "Weight"]
]

category_annotator = [['B', 'J', 'M'], ['B', 'E', 'J'],
                      ['E', 'J', 'N'], ['B', 'N', 'T'],
                      ['M', 'N', 'T'], ['E', 'N', 'S']]

category_id_dict = {"bags_and_cases": 0, "bluetooth": 1, "boots": 2, "keyboards": 3, "tv": 4, "vacuums": 5}

category_name_dict = {"bags_and_cases": ["bag", "case"], \
                      "bluetooth": ["bluetooth"], \
                      "boots": ["boot"], \
                      "keyboards": ["keyboard"], \
                      "tv": ["tv", "television"], \
                      "vacuums": ["vacuum"]}

id_category_dict = {v: k for k, v in category_id_dict.items()}

opinion_words = []
with open('data/opinion-lexicon-English/positive-words.txt', 'r') as fr:
    words = fr.read().replace('\n', ' ').split()
    opinion_words += words
with open('data/opinion-lexicon-English/negative-words.txt', 'r', encoding="utf-8") as fr:
    words = fr.read().replace('\n', ' ').strip().split()
    opinion_words += words
opinion_words = set(opinion_words)

from nltk.corpus import stopwords
stop_list = ["the", "be", "and", "of", "a", "in", "to", "have", "to", "it",
                 "I", "that", "for", "you", "he", "with", "on", "do", "say", "this", "they",
                 "at", "but", "we", "his", "from", "that", "not", "by", "she", "or",
                 "as", "what", "go", "their", "can", "who", "get", "if", "would", "her", "all",
                 "my", "make", "about", "know", "will", "as", "up", "one", "time", "there", "year",
                 "so", "think", "when", "which", "them", "some", "me", "people", "take", "out",
                 "into", "just", "see", "him", "your", "come", "could", "now", "than", "like",
                 "other", "how", "then", "its", "our", "two", "more", "these", "want", "way",
                 "look", "first", "also", "new", "because", "day", "more", "use", "no", "man",
                 "find", "here", "thing", "give", "many", "well", "1a", "b", "bb", "bbb", "x"]
stopsets = set(stopwords.words('english')) | set(stop_list)


def load_salience_dict(salience_path):
    sent_salience_dict = {}
    with open(salience_path, 'r') as fr:
        for line in fr:
            line = line.strip()
            if not len(line):
                continue
            scode, salience_scores = line.split('\t', 1)
            salience_scores = list(map(int, salience_scores.split()))
            sent_salience_dict[scode] = salience_scores
    return sent_salience_dict


def f_score(true_matrix, pred_matrix):
    tp = np.sum(np.logical_and(true_matrix, pred_matrix) * 1, axis=0)
    fp = np.sum(np.logical_and(true_matrix == 0, pred_matrix == 1) * 1, axis=0)
    fn = np.sum(np.logical_and(true_matrix == 1, pred_matrix == 0) * 1, axis=0)

    p = 1.0 * np.sum(tp) / (np.sum(tp) + np.sum(fp))
    r = 1.0 * np.sum(tp) / (np.sum(tp) + np.sum(fn))
    F = 2 * (p * r) / (p + r)
    return F


def evaluate(true_labels, pred_labels, category, eval_mode='multi_label'):
    classes = category_aspects[category_id_dict[category]]
    class_count = len(classes)
    none_idx = classes.index('none')
    instance_count = len(true_labels)
    assert len(true_labels) == len(pred_labels)

    true_label_matrix = np.zeros((instance_count, class_count))
    pred_label_matrix = np.zeros((instance_count, class_count))
    for i in range(instance_count):
        if pred_labels[i] >= class_count:
            pred_label_matrix[i][none_idx] = 1
        else:
            pred_label_matrix[i][pred_labels[i]] = 1
        for j in true_labels[i]:
            true_label_matrix[i][j] = 1

    f1 = f1_score(true_label_matrix, pred_label_matrix, average='micro')
    return f1


def is_opinion(sent):
    # sent_words = set(sent.split())
    doc = nlp(sent)
    sent_words = set([token.lemma_ for i, token in enumerate(doc)])
    if len(sent_words & opinion_words):
        return 1
    return 0


def get_sentiment(category):
    scode_sentiment_dict = {}
    with open('./sentiment/review_test_score_{}.txt'.format(category)) as fr:
        for line in fr:
            line = line.strip()
            if not len(line):
                continue
            senti_score, scode, original_text = line.split('\t', 2)
            scode_sentiment_dict[scode] = float(senti_score)
    return scode_sentiment_dict


def get_mate():
    scode_mate_dict = {}
    with open('./out/asp_prob_diff.txt') as fr:
        for line in fr:
            line = line.strip()
            if not len(line):
                continue
            scode, mate_score = line.split('\t', 1)
            scode_mate_dict[scode] = float(mate_score)
    return scode_mate_dict


def get_aspect_memory_norm(seeds_file, word2id, asp_cnt, category, sim_thres):
    none_idx = category_aspects[category_id_dict[category]].index('none')
    with open(seeds_file, 'r') as fr:
        seeds_ids_whole = []
        seeds_weights_whole = []
        for line_idx, line in enumerate(fr):
            if line_idx == none_idx:
                continue
            seeds_ids = []
            seeds_weights = []
            for tok in line.split():
                word, weight = tok.split(':')
                if word in word2id and word2id[word] not in seeds_ids:
                    seeds_ids.append(word2id[word])
                    seeds_weights.append(float(weight))
                    if len(seeds_ids) > asp_cnt:
                        break

            for wid, w_weight in zip(seeds_ids, seeds_weights):
                try:
                    w_idx = seeds_ids_whole.index(wid)
                    seeds_weights_whole[w_idx] = max(seeds_weights_whole[w_idx], w_weight)
                except:
                    seeds_ids_whole.append(wid)
                    seeds_weights_whole.append(w_weight)
    seeds_weights_whole = normalize_array(seeds_weights_whole, sim_thres * 1, 1)
    seeds_ids_whole, seeds_weights_whole = \
        (list(t) for t in zip(*sorted(zip(seeds_ids_whole, seeds_weights_whole), key=lambda x: x[1], reverse=True)))
    return seeds_ids_whole[:asp_cnt], seeds_weights_whole[:asp_cnt]


def clean_str(string):
    def replace_zero(matched):
        value = len(matched.group().strip().split('.')[0])
        return ' {} '.format('b' * value)

    string = string.lower()
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"&#34;", " ", string)
    string = re.sub(r"(http://)?www\.[^ ]+", " url ", string)

    string = re.sub('(?P<value>\s[0-9\.]+)\s', replace_zero, string)
    string = re.sub(r"\S*[0-9]+\S*", "1a", string)

    string = re.sub(r"[^a-z0-9$\'_]", " ", string)
    string = re.sub(r"_{2,}", "_", string)
    string = re.sub(r"\$+", " $ ", string)
    string = re.sub(r"rrb", " ", string)
    string = re.sub(r"lrb", " ", string)
    string = re.sub(r"rsb", " ", string)
    string = re.sub(r"lsb", " ", string)
    string = re.sub(r"(?<=[a-z])I", " I", string)

    string = re.sub(r"(?<= )[0-9]+(?= )", "NUM", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def normalize_array(a, a_min=0, a_max=1):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(a_min, a_max))
    a = np.expand_dims(np.array(a), axis=1)
    a = min_max_scaler.fit_transform(a)
    a = np.squeeze(a)
    return a


def sort_summary(sys_sum):
    sys_sum2 = []
    for i, sent in enumerate(sys_sum):
        if i == 0:
            sys_sum2.append(sent)
        else:
            if sent[1] == sys_sum2[-1][1] + 1:
                new_sent = (sys_sum2[-1][0] + ' ' + sent[0], sent[1], sys_sum2[-1][2] + sent[2], sent[3])
                sys_sum2[-1] = new_sent
            else:
                sys_sum2.append(sent)
    sys_sum_sorted = sorted(sys_sum2, key=lambda tup: (-tup[3], tup[1]))
    system_sum_content = '\n'.join([sent[0] for sent in sys_sum_sorted])
    return system_sum_content
