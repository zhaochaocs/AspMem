#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @author: Chao  
 @contact: zhaochaocs@gmail.com  
 @time: 1/20/2019 10:00 PM
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import AspectMemorySummarizer
from data_utils import batch_generator
from ILP_solver import ILPSolve
from utils import get_aspect_memory_norm, get_sentiment, is_opinion, sort_summary

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--category', help="category name to be summarized", type=str)
    parser.add_argument('--min_len', help="Minimum number of non-stop-words in segment (default: 2)", type=int,
                        default=2)
    parser.add_argument('--seeds', help='file that contains aspect seed words (auto/mate/gold)',
                        type=str, default='auto')
    parser.add_argument('--aspect_num', help="number of seed words", type=int,
                        default=100)
    parser.add_argument('--sim_thres', help="sim threshold of words", type=float, default=0.3)

    parser.add_argument('--score', help="score mode(default/norep/norel/nosenti/mate)", type=str, default='default')
    parser.add_argument('--opt', help='optimazation method (greedy/ILP)',
                        type=str, default='ILP')
    parser.add_argument('--redudency_filtering', help="redudency_filtering", action='store_true')    
    parser.add_argument('--ILP_num', help="number of ILP candidates", type=int,
                        default=20)
    parser.add_argument('--out_path', help="summary output path", type=str, default='./out/system_summary')
    parser.add_argument('--res_name', help="summary output path", type=str, default='')
    parser.add_argument('-q', '--quiet', help="No information to stdout", action='store_true')

    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    assert len(args.res_name) > 0

    # load tow dicts
    id2word = {}
    word2id = {}
    with open(args.data + '_word_mapping.txt', 'r') as fvoc:
        for line in fvoc:
            word, id = line.split()
            id2word[int(id)] = word
            word2id[word] = int(id)

    # load the data
    with open(args.data + '.pkl', 'rb') as frb:
        dataset = pickle.load(frb)
    with open(args.data + '_DEV.pkl', 'rb') as frb:
        dataset_dev = pickle.load(frb)
    with open(args.data + '_TEST.pkl', 'rb') as frb:
        dataset_sum = pickle.load(frb)
    # dataset_sum = {'data': dataset_dev['data'] + dataset_test['data'],
    #                'data_pos': dataset_dev['data_pos'] + dataset_test['data_pos'],
    #                "scodes": dataset_dev['scodes'] + dataset_test['scodes'],
    #                "original": dataset_dev['original'] + dataset_test['original']}

    w_emb_array = dataset['w2v']
    w_emb = torch.from_numpy(w_emb_array)
    vocab_size, emb_size = w_emb.size()

    # load aspect info
    args.aspect_seeds = 'data/preprocessed/aspects/{}_keywords.txt'.format(args.category)
    a_id, a_weight = get_aspect_memory_norm(args.aspect_seeds, word2id, args.aspect_num, args.category, args.sim_thres)
    a_words = [id2word[id_] for id_ in a_id]
    a_emb = torch.from_numpy(np.array([w_emb_array[seeds] for seeds in a_id], dtype='float32'))
    print("Overwrite the number of aspects as {}".format(len(a_weight)))

    # for each product, pick up the reviews of this product
    prodID_reviewID = {}
    for i, scode in enumerate(dataset_sum['scodes']):
        prod_code = scode.split('-', 1)[0]
        if prod_code not in prodID_reviewID:
            prodID_reviewID[prod_code] = [i]
        else:
            prodID_reviewID[prod_code] += [i]

    scodes_id_dict = {}
    with open('data/gold/salience/{}.sal'.format(args.category)) as fr:
        for line in fr:
            scode, _ = line.split('\t')
            scodes_id_dict[scode] = len(scodes_id_dict)

    scodes_original_dict = {}
    for sc, orig in zip(dataset_sum['scodes'], dataset_sum['original']):
        scodes_original_dict[sc] = orig
    original_scodes_dict = {v: k for k, v in scodes_original_dict.items()}

    scode_aspect_dict = {scode: aspect for scode, aspect in zip(dataset_sum['scodes'], dataset_sum['labels'])}

    # load the sentiment score
    scode_sentiment_dict = get_sentiment(args.category)

    # get review representation
    net = AspectMemorySummarizer(vocab_size, emb_size,
                                 w_emb=w_emb, a_emb=a_emb, a_weight=a_weight, word_thres=args.sim_thres)
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    asp_prob = []
    sent_rep, sent_score = [], []

    batch_size = 100
    num_batches_per_epoch = int((len(dataset_sum['data']) - 1) / batch_size) + 1
    batched_data = batch_generator(dataset_sum, batch_size, shuffle=False)

    with tqdm(total=num_batches_per_epoch) as pbar:
        for inputs, inputs_original in batched_data:
            pbar.update(1)

            if inputs.shape[1] < args.min_len:
                continue
            if torch.cuda.is_available():
                inputs = inputs.cuda()  # b*s

            with torch.no_grad():
                enc_out, attention, score = net(inputs)  # attention: b*s  # positives: batch_size * emb_size
            sent_rep.append(enc_out)
            sent_score.append(score)  # b

            centroid_score, keyword_idx = net.get_attention_info()

    sent_rep = torch.cat(sent_rep, dim=0)
    sent_score = torch.cat(sent_score, dim=0)

    # get filename of golden products
    prod_set = set()
    for filename in os.listdir('data/gold/summaries/{}/all'.format(args.category)):
        prod_id = filename.split('.')[1]
        prod_set.add(prod_id)
    prod_set = list(prod_set)
    prod_set.sort()

    system_summary = []

    for prod in prod_set:
        sys_sum = []

        review_idx = prodID_reviewID[prod]
        review_idx = torch.tensor(review_idx).cuda()
        candidates_rep = sent_rep.index_select(dim=0, index=review_idx)  # n * emb
        relate_score = sent_score.index_select(dim=0, index=review_idx).data.cpu().numpy()
        senti_score = np.array([scode_sentiment_dict[dataset_sum['scodes'][global_idx]] for global_idx in review_idx])

        sum_df = pd.DataFrame(
            columns=['idx', 'relateness', 'sentiment', 'scode', 'text'])
        for local_idx, global_idx in enumerate(review_idx):  # for each k
            sum_df.loc[local_idx] = [
                global_idx,
                relate_score[local_idx],
                scode_sentiment_dict[dataset_sum['scodes'][global_idx]],
                dataset_sum['scodes'][global_idx],
                dataset_sum['original'][global_idx]]

        sum_df['sentiment_flag'] = sum_df.apply(lambda x: np.sign(x['sentiment']), axis=1)
        sum_df['sentiment'] = sum_df.apply(lambda x: np.abs(x['sentiment']), axis=1)
        sum_df['opinion'] = sum_df.apply(lambda x: is_opinion(x['text']), axis=1)
        sum_df['score'] = sum_df.apply(
            lambda x: x['relateness'] * x['sentiment'] * x['opinion'], axis=1)

        sum_len = 0
        sum_idx = []
        sum_sorted_df = sum_df.sort_values(by=['score'], ascending=False)
        sim_thres = nn.Threshold(0.5, 0)

        def select_opinion(candidate_size, _lambda=0):
            s, idx_list, l, senti = [], [], [], []
            for i in range(0, candidate_size):
                selected_text = sum_sorted_df.head(i + 1).iloc[-1]['text']
                selected_idx = sum_sorted_df.head(i + 1).iloc[-1]['idx']
                senti.append(sum_sorted_df.head(i + 1).iloc[-1]['sentiment_flag'])
                if len(selected_text.split(' ')) > 20:
                    current_score = 0
                else:
                    current_score = sum_sorted_df.head(i + 1).iloc[-1]['score']
                idx_list.append(selected_idx)
                l.append(len(selected_text.split(' ')))
                s.append(current_score)
            idx_list = torch.stack(idx_list)
            candidates_rep = sent_rep.index_select(dim=0, index=idx_list)
            u1 = candidates_rep.unsqueeze(0)
            u2 = candidates_rep.unsqueeze(1)
            sim = F.cosine_similarity(u1, u2, dim=-1)  # n*n
            d = (sim > 0.5).float() - torch.eye(candidate_size).float().cuda()
            d = d.data.cpu().numpy().tolist()

            select = ILPSolve(s, d, l, _lambda)
            return select

        sum_sorted_df.insert(1, 'select', '0')
        candidate_size = args.ILP_num
        select = select_opinion(candidate_size, _lambda=100)
        for i in range(0, candidate_size):
            if select[i]:
                sum_sorted_df.iloc[i, sum_sorted_df.columns.get_loc('select')] = 1
                selected_text = sum_sorted_df.head(i + 1).iloc[-1]['text']
                selected_senti_sign = sum_sorted_df.head(i + 1).iloc[-1]['sentiment_flag']
                sys_sum.append((selected_text,
                                scodes_id_dict[original_scodes_dict[selected_text]],
                                scode_aspect_dict[original_scodes_dict[selected_text]],
                                selected_senti_sign))


        sequence = ['idx', 'select', 'score', 'relateness', 'sentiment', 'opinion', 'text']
        sum_sorted_df = sum_sorted_df.reindex(columns=sequence)
        system_sum_content = sort_summary(sys_sum)

